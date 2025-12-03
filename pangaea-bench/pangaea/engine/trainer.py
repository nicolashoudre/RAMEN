import copy
import logging
import operator
import os
import pathlib
import time
import numpy as np
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Subset
from pangaea.utils.logger import RunningAverageMeter, sec_to_hm

from torch.profiler import profile, record_function, ProfilerActivity
from fvcore.nn import FlopCountAnalysis, parameter_count_table

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module | None,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        evaluator: torch.nn.Module,
        n_epochs: int,
        exp_dir: pathlib.Path | str,
        device: torch.device,
        precision: str,
        use_wandb: bool,
        ckpt_interval: int,
        eval_interval: int,
        log_interval: int,
        best_metric_key: str,
    ):
        """Initialize the Trainer.

        Args:
            model (nn.Module): model to train (encoder + decoder).
            train_loader (DataLoader): train data loader.
            criterion (nn.Module): criterion to compute the loss.
            optimizer (Optimizer): optimizer to update the model's parameters.
            lr_scheduler (LRScheduler): lr scheduler to update the learning rate.
            evaluator (torch.nn.Module): task evaluator to evaluate the model.
            n_epochs (int): number of epochs to train the model.
            exp_dir (pathlib.Path | str): path to the experiment directory.
            device (torch.device): model
            precision (str): precision to train the model (fp32, fp16, bfp16).
            use_wandb (bool): whether to use wandb for logging.
            ckpt_interval (int): interval to save the checkpoint.
            eval_interval (int): interval to evaluate the model.
            log_interval (int): interval to log the training information.
            best_metric_key (str): metric that determines best checkpoints.
        """
        self.rank = int(os.environ["RANK"])
        self.criterion = criterion
        self.model = model
        self.train_loader = train_loader
        self.batch_per_epoch = len(self.train_loader)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator
        self.n_epochs = n_epochs
        self.logger = logging.getLogger()
        self.exp_dir = exp_dir
        self.device = device
        self.use_wandb = use_wandb
        self.ckpt_interval = ckpt_interval
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.best_metric_key = best_metric_key

        self.training_stats = {
            name: RunningAverageMeter(length=self.batch_per_epoch)
            for name in ["loss", "data_time", "batch_time", "eval_time"]
        }
        self.training_metrics = {}
        self.best_metric_comp = operator.gt
        self.num_classes = self.train_loader.dataset.num_classes

        assert precision in [
            "fp32",
            "fp16",
            "bfp16",
        ], f"Invalid precision {precision}, use 'fp32', 'fp16' or 'bfp16'."
        self.enable_mixed_precision = precision != "fp32"
        self.precision = torch.float16 if (precision == "fp16") else torch.bfloat16
        # self.scaler = torch.GradScaler("cuda", enabled=self.enable_mixed_precision)
        self.scaler = torch.cuda.amp.GradScaler("cuda", enabled=self.enable_mixed_precision)

        self.start_epoch = 0

        if self.use_wandb:
            import wandb

            self.wandb = wandb

    def train(self) -> None:
        """Train the model for n_epochs then evaluate the model and save the best model."""
        # end_time = time.time()
        for epoch in range(self.start_epoch, self.n_epochs):
            # train the network for one epoch
            if epoch % self.eval_interval == 0:
                metrics, used_time = self.evaluator(self.model, f"epoch {epoch}")
                self.training_stats["eval_time"].update(used_time)
                self.save_best_checkpoint(metrics, epoch)

            self.logger.info("============ Starting epoch %i ... ============" % epoch)
            # set sampler
            self.t = time.time()
            self.train_loader.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)
            if epoch % self.ckpt_interval == 0 and epoch != self.start_epoch:
                self.save_model(epoch)

        metrics, used_time = self.evaluator(self.model, "final model")
        self.training_stats["eval_time"].update(used_time)
        self.save_best_checkpoint(metrics, self.n_epochs)

        # save last model
        self.save_model(self.n_epochs, is_final=True)

    def train_one_epoch(self, epoch: int) -> None:
        """Train model for one epoch.

        Args:
            epoch (int): number of the epoch.
        """
        self.model.train()

        end_time = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            image, target = data["image"], data["target"]
            image = {modality: value.to(self.device) for modality, value in image.items()}
            target = target.to(self.device)

            self.training_stats["data_time"].update(time.time() - end_time)

            with torch.autocast(
                "cuda", enabled=self.enable_mixed_precision, dtype=self.precision
            ):
                logits = self.model(image, output_shape=target.shape[-2:])
                loss = self.compute_loss(logits, target)

            self.optimizer.zero_grad()

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Rank {self.rank} got infinite/NaN loss at batch {batch_idx} of epoch {epoch}!"
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.training_stats['loss'].update(loss.item())
            with torch.no_grad():
                self.compute_logging_metrics(logits, target)
            if (batch_idx + 1) % self.log_interval == 0:
                self.log(batch_idx + 1, epoch)

            self.lr_scheduler.step()

            if self.use_wandb and self.rank == 0:
                self.wandb.log(
                    {
                        "train_loss": loss.item(),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        **{
                            f"train_{k}": v.avg
                            for k, v in self.training_metrics.items()
                        },
                    },
                    step=epoch * len(self.train_loader) + batch_idx,
                )

            self.training_stats["batch_time"].update(time.time() - end_time)
            end_time = time.time()

    def get_checkpoint(self, epoch: int) -> dict[str, dict | int]:
        """Create a checkpoint dictionary, containing references to the pytorch tensors.

        Args:
            epoch (int): number of the epoch.

        Returns:
            dict[str, dict | int]: checkpoint dictionary.
        """
        checkpoint = {
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
        }
        return checkpoint

    def save_model(
        self,
        epoch: int,
        is_final: bool = False,
        is_best: bool = False,
        checkpoint: dict[str, dict | int] | None = None,
    ):
        """Save the model checkpoint.

        Args:
            epoch (int): number of the epoch.
            is_final (bool, optional): whether is the final checkpoint. Defaults to False.
            is_best (bool, optional): wheter is the best checkpoint. Defaults to False.
            checkpoint (dict[str, dict  |  int] | None, optional): already prepared checkpoint dict. Defaults to None.
        """
        if self.rank != 0:
            torch.distributed.barrier()
            return
        checkpoint = self.get_checkpoint(epoch) if checkpoint is None else checkpoint
        suffix = "_best" if is_best else f"{epoch}_final" if is_final else f"{epoch}"
        checkpoint_path = os.path.join(self.exp_dir, f"checkpoint_{suffix}.pth")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(
            f"Epoch {epoch} | Training checkpoint saved at {checkpoint_path}"
        )
        torch.distributed.barrier()
        return

    def load_model(self, resume_path: str | pathlib.Path) -> None:
        """Load model from the checkpoint.

        Args:
            resume_path (str | pathlib.Path): path to the checkpoint.
        """
        model_dict = torch.load(resume_path, map_location=self.device, weights_only=False)
        if "model" in model_dict:
            self.model.module.load_state_dict(model_dict["model"])
            self.optimizer.load_state_dict(model_dict["optimizer"])
            self.lr_scheduler.load_state_dict(model_dict["lr_scheduler"])
            self.scaler.load_state_dict(model_dict["scaler"])
            self.start_epoch = model_dict["epoch"] + 1
        else:
            self.model.module.load_state_dict(model_dict)
            self.start_epoch = 0

        self.logger.info(
            f"Loaded model from {resume_path}. Resume training from epoch {self.start_epoch}"
        )

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits (torch.Tensor): logits from the model.
            target (torch.Tensor): target tensor.

        Raises:
            NotImplementedError: raise if the method is not implemented.

        Returns:
            torch.Tensor: loss value.
        """
        raise NotImplementedError

    def save_best_checkpoint(
        self, eval_metrics: dict[float, list[float]], epoch: int
    ) -> None:
        """Update the best checkpoint according to the evaluation metrics.

        Args:
            eval_metrics (dict[float, list[float]]): metrics computed by the evaluator on the validation set.
            epoch (int): number of the epoch.
        """
        curr_metric = eval_metrics[self.best_metric_key]
        if isinstance(curr_metric, list):
            curr_metric = curr_metric[1] if self.num_classes == 1 else np.mean(curr_metric)
        if self.best_metric_comp(curr_metric, self.best_metric):
            self.best_metric = curr_metric
            best_ckpt = self.get_checkpoint(epoch)
            self.save_model(
                epoch, is_best=True, checkpoint=best_ckpt
            )

    @torch.no_grad()
    def compute_logging_metrics(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> dict[float, list[float]]:
        """Compute logging metrics.

        Args:
            logits (torch.Tensor): logits output by the decoder.
            target (torch.Tensor): target tensor.

        Raises:
            NotImplementedError: raise if the method is not implemented.

        Returns:
            dict[float, list[float]]: logging metrics.
        """
        raise NotImplementedError

    def log(self, batch_idx: int, epoch) -> None:
        """Log the information.

        Args:
            batch_idx (int): number of the batch.
            epoch (_type_): number of the epoch.
        """
        # TO DO: upload to wandb
        left_batch_this_epoch = self.batch_per_epoch - batch_idx
        left_batch_all = (
            self.batch_per_epoch * (self.n_epochs - epoch - 1) + left_batch_this_epoch
        )
        left_eval_times = ((self.n_epochs - 0.5) // self.eval_interval + 2
                           - self.training_stats["eval_time"].count)
        left_time_this_epoch = sec_to_hm(
            left_batch_this_epoch * self.training_stats["batch_time"].avg
        )
        left_time_all = sec_to_hm(
            left_batch_all * self.training_stats["batch_time"].avg
            + left_eval_times * self.training_stats["eval_time"].avg
        )

        basic_info = (
            "Epoch [{epoch}-{batch_idx}/{len_loader}]\t"
            "ETA [{left_time_all}|{left_time_this_epoch}]\t"
            "Time [{batch_time.avg:.3f}|{data_time.avg:.3f}]\t"
            "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "lr {lr:.3e}".format(
                epoch=epoch,
                len_loader=len(self.train_loader),
                batch_idx=batch_idx,
                left_time_this_epoch=left_time_this_epoch,
                left_time_all=left_time_all,
                batch_time=self.training_stats["batch_time"],
                data_time=self.training_stats["data_time"],
                loss=self.training_stats["loss"],
                lr=self.optimizer.param_groups[0]["lr"],
            )
        )

        metrics_info = [
            "{} {:>7} ({:>7})".format(k, "%.3f" % v.val, "%.3f" % v.avg)
            for k, v in self.training_metrics.items()
        ]
        metrics_info = "\n Training metrics: " + "\t".join(metrics_info)
        # extra_metrics_info = self.extra_info_template.format(**self.extra_info)
        log_info = basic_info + metrics_info
        self.logger.info(log_info)

    def reset_stats(self) -> None:
        """Reset the training stats and metrics."""
        for v in self.training_stats.values():
            v.reset()
        for v in self.training_metrics.values():
            v.reset()


class LinearClassificationTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        evaluator: torch.nn.Module,
        n_epochs: int,
        exp_dir: pathlib.Path | str,
        device: torch.device,
        precision: str,
        use_wandb: bool,
        ckpt_interval: int,
        eval_interval: int,
        log_interval: int,
        best_metric_key: str,
        multi_label: bool = False,  # <-- Flag for multi-label classification, e.g., BigEarthNet dataset
        topk: int = 1,  # Top-k predictions to use in multi-label scenario
    ):
        """Initialize the Trainer for Classification task.

        Args:
            model (nn.Module): model to train (encoder + decoder).
            train_loader (DataLoader): train data loader.
            criterion (nn.Module): criterion to compute the loss.
            optimizer (Optimizer): optimizer to update the model's parameters.
            lr_scheduler (LRScheduler): lr scheduler to update the learning rate.
            evaluator (torch.nn.Module): task evaluator to evaluate the model.
            n_epochs (int): number of epochs to train the model.
            exp_dir (pathlib.Path | str): path to the experiment directory.
            device (torch.device): model
            precision (str): precision to train the model (fp32, fp16, bfp16).
            use_wandb (bool): whether to use wandb for logging.
            ckpt_interval (int): interval to save the checkpoint.
            eval_interval (int): interval to evaluate the model.
            log_interval (int): interval to log the training information.
            best_metric_key (str): metric that determines best checkpoints.
            multi_label (bool): Flag to enable multi-label classification.
        """
        super().__init__(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            evaluator=evaluator,
            n_epochs=n_epochs,
            exp_dir=exp_dir,
            device=device,
            precision=precision,
            use_wandb=use_wandb,
            ckpt_interval=ckpt_interval,
            eval_interval=eval_interval,
            log_interval=log_interval,
            best_metric_key=best_metric_key,
        )
        
        self.multi_label = multi_label
        self.topk = topk

        self.training_metrics = {
            name: RunningAverageMeter(length=100) for name in ["accuracy", "F1"]
        }
        self.best_metric = float("-inf")
        self.best_metric_comp = operator.gt

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        return self.criterion(logits, target)

    def compute_logging_metrics(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Compute logging metrics.
        For multi-label:
        - Uses sigmoid activation and top-k selection.
        For single-class:
        - Uses argmax and converts predictions to one-hot encoding.

        Args:
            logits (torch.Tensor): logits from the decoder.
            target (torch.Tensor): target tensor.
        """
        if self.multi_label:
            preds_prob = torch.sigmoid(logits)
            topk_indices = preds_prob.topk(self.topk, dim=1).indices  
            preds = torch.zeros_like(preds_prob, dtype=torch.bool)
            preds.scatter_(1, topk_indices, 1)
        else:
            preds = torch.argmax(logits, dim=1)
            
            one_hot_preds = torch.zeros(
                size=(preds.size(0), self.num_classes),
                device=preds.device,
                dtype=torch.bool
            )
            one_hot_preds.scatter_(1, preds.unsqueeze(1), 1)
            preds = one_hot_preds
            # Convert targets to one-hot.
            one_hot_targets = torch.zeros_like(preds)
            one_hot_targets.scatter_(1, targets.unsqueeze(1), 1)
            targets = one_hot_targets
        
        # Micro-average: aggregate across all classes.
        preds = preds.bool()
        targets = targets.bool()
        TP = (preds & targets).sum().float()
        FP = (preds & ~targets).sum().float()
        FN = (~preds & targets).sum().float()
        TN = (~preds & ~targets).sum().float()
        
        acc = (TP + TN) / (TP + TN + FP + FN + 1e-8) 
        precision = TP / (TP + FP + 1e-8) 
        recall = TP / (TP + FN + 1e-8)  
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        self.training_metrics["accuracy"].update(acc.item())
        self.training_metrics["F1"].update(f1.item())
    

class KNNTrainer(Trainer):
    """A zero-learning shell so run.py can stay unchanged."""
    
    def __init__(
        self,
        model: nn.Module,              # should be KNNClassifier
        train_loader: DataLoader,
        evaluator,
        lr_scheduler,
        optimizer,
        criterion,
        exp_dir: pathlib.Path | str,
        device: torch.device,
        n_epochs: int,
        precision: str,
        use_wandb: bool,
    ):
        dummy_opt   = torch.optim.SGD([torch.empty(0, device=device, requires_grad=True)], lr=1)
        dummy_sched = torch.optim.lr_scheduler.LambdaLR(dummy_opt, lambda _: 1)

        super().__init__(
            model=model,
            train_loader=train_loader,
            criterion=nn.Identity(),       # never used
            optimizer=dummy_opt,
            lr_scheduler=dummy_sched,
            evaluator=evaluator,
            n_epochs=n_epochs,
            exp_dir=exp_dir,
            device=device,
            precision=precision,
            use_wandb=use_wandb,
            ckpt_interval=999,
            eval_interval=1,
            log_interval=999,
            best_metric_key="top1",
        )
        self.logger: logging.Logger = logging.getLogger()
        self.train_loader = train_loader
    # ------------------------------------------------------------------ #
    def train(self):
        self.logger.info("=========== k-NN evaluation only ===========")
        self.evaluator(self.model, model_name="probe", train_loader=self.train_loader)
        self.logger.info("============================================")
        dummy_path1 = os.path.join(self.exp_dir, "checkpoint_dummy_best.pth")
        dummy_path2 = os.path.join(self.exp_dir, "checkpoint_dummy_final.pth")
        if self.rank == 0 and not os.path.exists(dummy_path1):
            torch.save({"knn_probe": True}, dummy_path1)
        if self.rank == 0 and not os.path.exists(dummy_path2):
            torch.save({"knn_probe": True}, dummy_path2)

    # never called
    def compute_loss(self, logits, target): ...
    def compute_logging_metrics(self, logits, target): ...

                       
class SegTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        evaluator: torch.nn.Module,
        n_epochs: int,
        exp_dir: pathlib.Path | str,
        device: torch.device,
        precision: str,
        use_wandb: bool,
        ckpt_interval: int,
        eval_interval: int,
        log_interval: int,
        best_metric_key: str,
    ):
        """Initialize the Trainer for segmentation task.
        Args:
            model (nn.Module): model to train (encoder + decoder).
            train_loader (DataLoader): train data loader.
            criterion (nn.Module): criterion to compute the loss.
            optimizer (Optimizer): optimizer to update the model's parameters.
            lr_scheduler (LRScheduler): lr scheduler to update the learning rate.
            evaluator (torch.nn.Module): task evaluator to evaluate the model.
            n_epochs (int): number of epochs to train the model.
            exp_dir (pathlib.Path | str): path to the experiment directory.
            device (torch.device): model
            precision (str): precision to train the model (fp32, fp16, bfp16).
            use_wandb (bool): whether to use wandb for logging.
            ckpt_interval (int): interval to save the checkpoint.
            eval_interval (int): interval to evaluate the model.
            log_interval (int): interval to log the training information.
            best_metric_key (str): metric that determines best checkpoints.
        """
        super().__init__(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            evaluator=evaluator,
            n_epochs=n_epochs,
            exp_dir=exp_dir,
            device=device,
            precision=precision,
            use_wandb=use_wandb,
            ckpt_interval=ckpt_interval,
            eval_interval=eval_interval,
            log_interval=log_interval,
            best_metric_key=best_metric_key,
        )

        self.training_metrics = {
            name: RunningAverageMeter(length=100) for name in ["Acc", "mAcc", "mIoU"]
        }
        self.best_metric = float("-inf")
        self.best_metric_comp = operator.gt

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits (torch.Tensor): logits from the decoder.
            target (torch.Tensor): target tensor.

        Returns:
            torch.Tensor: loss value.
        """
        return self.criterion(logits, target)

    @torch.no_grad()
    def compute_logging_metrics(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> None:
        """Compute logging metrics.

        Args:
            logits (torch.Tensor): loggits from the decoder.
            target (torch.Tensor): target tensor.
        """
        # logits = F.interpolate(logits, size=target.shape[1:], mode='bilinear')
        num_classes = logits.shape[1]
        if num_classes == 1:
            pred = (torch.sigmoid(logits) > 0.5).type(torch.int64)
        else:
            pred = torch.argmax(logits, dim=1, keepdim=True)
        target = target.unsqueeze(1)
        ignore_mask = target == self.train_loader.dataset.ignore_index
        target[ignore_mask] = 0
        ignore_mask = ignore_mask.expand(
            -1, num_classes if num_classes > 1 else 2, -1, -1
        )

        dims = list(logits.shape)
        if num_classes == 1:
            dims[1] = 2
        binary_pred = torch.zeros(dims, dtype=bool, device=self.device)
        binary_target = torch.zeros(dims, dtype=bool, device=self.device)
        binary_pred.scatter_(dim=1, index=pred, src=torch.ones_like(binary_pred))
        binary_target.scatter_(dim=1, index=target, src=torch.ones_like(binary_target))
        binary_pred[ignore_mask] = 0
        binary_target[ignore_mask] = 0

        intersection = torch.logical_and(binary_pred, binary_target)
        union = torch.logical_or(binary_pred, binary_target)

        acc = intersection.sum() / binary_target.sum() * 100
        macc = (
            torch.nanmean(
                intersection.sum(dim=(0, 2, 3)) / binary_target.sum(dim=(0, 2, 3))
            )
            * 100
        )
        miou = (
            torch.nanmean(intersection.sum(dim=(0, 2, 3)) / union.sum(dim=(0, 2, 3)))
            * 100
        )

        self.training_metrics["Acc"].update(acc.item())
        self.training_metrics["mAcc"].update(macc.item())
        self.training_metrics["mIoU"].update(miou.item())


class RegTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        evaluator: torch.nn.Module,
        n_epochs: int,
        exp_dir: pathlib.Path | str,
        device: torch.device,
        precision: str,
        use_wandb: bool,
        ckpt_interval: int,
        eval_interval: int,
        log_interval: int,
        best_metric_key: str,
    ):
        """Initialize the Trainer for regression task.
        Args:
            model (nn.Module): model to train (encoder + decoder).
            train_loader (DataLoader): train data loader.
            criterion (nn.Module): criterion to compute the loss.
            optimizer (Optimizer): optimizer to update the model's parameters.
            lr_scheduler (LRScheduler): lr scheduler to update the learning rate.
            evaluator (torch.nn.Module): task evaluator to evaluate the model.
            n_epochs (int): number of epochs to train the model.
            exp_dir (pathlib.Path | str): path to the experiment directory.
            device (torch.device): model
            precision (str): precision to train the model (fp32, fp16, bfp16).
            use_wandb (bool): whether to use wandb for logging.
            ckpt_interval (int): interval to save the checkpoint.
            eval_interval (int): interval to evaluate the model.
            log_interval (int): interval to log the training information.
            best_metric_key (str): metric that determines best checkpoints.
        """
        super().__init__(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            evaluator=evaluator,
            n_epochs=n_epochs,
            exp_dir=exp_dir,
            device=device,
            precision=precision,
            use_wandb=use_wandb,
            ckpt_interval=ckpt_interval,
            eval_interval=eval_interval,
            log_interval=log_interval,
            best_metric_key=best_metric_key,
        )

        self.training_metrics = {
            name: RunningAverageMeter(length=100) for name in ["MSE"]
        }
        self.best_metric = float("inf")
        self.best_metric_comp = operator.lt

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits (torch.Tensor): logits from the decoder.
            target (torch.Tensor): target tensor.

        Returns:
            torch.Tensor: loss value.
        """
        return self.criterion(logits.squeeze(dim=1), target)

    @torch.no_grad()
    def compute_logging_metrics(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> None:
        """Compute logging metrics.

        Args:
            logits (torch.Tensor): logits from the decoder.
            target (torch.Tensor): target tensor.
        """

        mse = F.mse_loss(logits.squeeze(dim=1), target)  
        self.training_metrics["MSE"].update(mse.item())

class ModelProfiler(Trainer):
    """
    ModelProfiler: FLOPs + Latency profiler using fvcore.
    Automatically adapts to sliding inference if input_size < tile size.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        evaluator: torch.nn.Module,
        n_epochs: int,
        exp_dir: pathlib.Path | str,
        device: torch.device,
        precision: str,
        use_wandb: bool,
        ckpt_interval: int,
        eval_interval: int,
        log_interval: int,
        best_metric_key: str,
        inference_mode: str = "auto",  # 'auto', 'sliding', or 'whole'
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            evaluator=evaluator,
            n_epochs=n_epochs,
            exp_dir=exp_dir,
            device=device,
            precision=precision,
            use_wandb=use_wandb,
            ckpt_interval=ckpt_interval,
            eval_interval=eval_interval,
            log_interval=log_interval,
            best_metric_key=best_metric_key,
        )

        self.inference_mode = inference_mode
        self.sliding_inference_batch = 8

    @staticmethod
    def sliding_inference(model, img, input_size, output_shape=None, stride=None, max_batch=None):
        img_copy = {k: v.clone() if "_dates" not in k else v for k, v in img.items()}
        b, c, t, height, width = img_copy[list(img_copy.keys())[0]].shape
    
        if stride is None:
            h = int(math.ceil(height / input_size))
            w = int(math.ceil(width / input_size))
        else:
            h = math.ceil((height - input_size) / stride) + 1
            w = math.ceil((width - input_size) / stride) + 1
    
        h_grid = torch.linspace(0, height - input_size, h).round().long()
        w_grid = torch.linspace(0, width - input_size, w).round().long()
        num_crops_per_img = h * w
    
        crops_dict = {}
        for k, v in img_copy.items():
            if "_dates" in k:
                crops_dict[k] = v
                continue
            crops = []
            for i in range(h):
                for j in range(w):
                    crops.append(v[:, :, :, h_grid[i]:h_grid[i] + input_size, w_grid[j]:w_grid[j] + input_size])
            crops_dict[k] = torch.cat(crops, dim=0)
    
        for k, v in crops_dict.items():
            if "_dates" in k:
                expanded_dates = v.unsqueeze(1).repeat(1, num_crops_per_img, 1).view(-1, v.shape[1])
                crops_dict[k] = expanded_dates

        pred_list = []
        max_batch = max_batch if max_batch is not None else b * num_crops_per_img
        batch_num = int(math.ceil(b * num_crops_per_img / max_batch))
        for i in range(batch_num):
            img_batch = {k: v[max_batch * i: min(max_batch * i + max_batch, b * num_crops_per_img)] 
                         for k, v in crops_dict.items()}
            pred_batch = model.forward(img_batch, output_shape=(input_size, input_size))
            pred_list.append(pred_batch)
        pred = torch.cat(pred_list, dim=0)
    
        pred = pred.view(num_crops_per_img, b, -1, input_size, input_size).transpose(0, 1)
        merged_pred = torch.zeros((b, pred.shape[2], height, width), device=pred.device)
        pred_count = torch.zeros((b, height, width), dtype=torch.long, device=pred.device)
    
        for i in range(h):
            for j in range(w):
                merged_pred[:, :, h_grid[i]:h_grid[i] + input_size, w_grid[j]:w_grid[j] + input_size] += pred[:, h * i + j]
                pred_count[:, h_grid[i]:h_grid[i] + input_size, w_grid[j]:w_grid[j] + input_size] += 1
    
        merged_pred = merged_pred / pred_count.unsqueeze(1)
        if output_shape is not None:
            merged_pred = F.interpolate(merged_pred, size=output_shape, mode="bilinear")
    
        return merged_pred, num_crops_per_img

    @torch.no_grad()
    def profile_model(
        self,
        model,
        warmup_steps: int = 32,
        measure_steps: int = 16,
        use_cuda: bool = True,
    ):
        """
        Profiles the model using fvcore for FLOPs.
        """
        device = self.device
        model = model.to(device).eval()
    
        # One sample batch
        example_batch = next(iter(self.train_loader))
        image = {k: v.to(device) for k, v in example_batch["image"].items()}
        target_shape = example_batch.get("target", None)
        if target_shape is not None:
            target_shape = target_shape.shape[-2:]
    
        # Determine mode
        input_size = model.module.encoder.input_size
        h, w = target_shape[0], target_shape[1]
        print(input_size)
        print(h)
        if self.inference_mode == "sliding" or (
            self.inference_mode == "auto" and (h > input_size or w > input_size)
        ):
            mode = "sliding"
        else:
            mode = "whole"
    
        self.logger.info(f"Using '{mode}' inference mode for profiling...")
    
        # Warmup
        self.logger.info(f"Warming up model for {warmup_steps} steps...")
        for _ in range(warmup_steps):
            if mode == "sliding":
                _, _ = self.sliding_inference(
                    model,
                    image,
                    input_size,
                    output_shape=target_shape,
                    max_batch=self.sliding_inference_batch,
                )
            else:
                _ = model(image) if target_shape is None else model(image, output_shape=target_shape)
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

        # Measure latency
        self.logger.info(f"⚙️ Measuring latency over {measure_steps} steps...")
        start_time = time.time()
        for _ in range(measure_steps):
            if mode == "sliding":
                _, _ = self.sliding_inference(
                    model,
                    image,
                    input_size,
                    output_shape=target_shape,
                    max_batch=self.sliding_inference_batch,
                )
            else:
                _ = model(image) if target_shape is None else model(image, output_shape=target_shape)
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
        total_time = time.time() - start_time
        avg_latency_ms = (total_time / measure_steps) * 1000
    
        # FLOPs + Params
        self.logger.info("Computing FLOPs and Params via fvcore...")
        try:
            total_params = sum(p.numel() for p in model.parameters())
    
            if mode == "sliding":
                # Compute how many crops we do per full image
                _, num_crops_per_img = self.sliding_inference(
                    model,
                    image,
                    input_size,
                    output_shape=target_shape,
                    max_batch=16,
                )
                # Use fvcore on a single crop
                self.logger.info(f"Num crops per tile:{num_crops_per_img}")
                sample_crop = {
                    k: v[:, :, :, :input_size, :input_size] if "_dates" not in k else v
                    for k, v in image.items()
                }
                flops = FlopCountAnalysis(model, (sample_crop,))
                total_flops = flops.total() * num_crops_per_img
            else:
                flops = FlopCountAnalysis(model, (image,))
                total_flops = flops.total()

            gflops = total_flops / 1e9
            gflops_per_sec = gflops / (avg_latency_ms / 1000) if avg_latency_ms > 0 else 0
    
        except Exception as e:
            self.logger.warning(f"⚠️ fvcore FLOPs analysis failed: {e}")
            total_params = sum(p.numel() for p in model.parameters())
            gflops = 0.0
            gflops_per_sec = 0.0
    
        results = {
            "Params (M)": total_params / 1e6,
            "FLOPs (G)": gflops,
            "Avg Latency (ms)": avg_latency_ms,
            "Throughput (GFLOPs/s)": gflops_per_sec,
        }
    
        self.log_profile_results(results)
        return results
    
    def train(self):
        """Run the profiling directly when .train() is called (for compatibility)."""
        return self.profile_model(self.model, warmup_steps=20, measure_steps=10, use_cuda=True)
    
    def log_profile_results(self, results):
        self.logger.info("\n Model Profiling Summary")
        self.logger.info("===============================")
        for k, v in results.items():
            self.logger.info(f"{k:25s}: {v:.3f}")
        self.logger.info("===============================\n")
    
        if self.use_wandb and getattr(self, "rank", 0) == 0:
            import wandb
            wandb.log({f"profile/{k}": v for k, v in results.items()})