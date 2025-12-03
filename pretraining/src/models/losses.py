import torch
from torch import nn
import torch.nn.functional as F
import einops
from einops import rearrange
from models.utils import FullGatherLayer

class CrossEntropyWeighted(nn.Module):
    def __init__(self, num_classes):
        super(CrossEntropyWeighted, self).__init__()
        self.weights = torch.ones(num_classes).float()
        self.weights[-1] = 0

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor BxN that contains the logits
            y: dict that contains "label": torch.Tensor BxN
        Returns:
            torch.Tensor: CrossEntropy loss between x and y: torch.Tensor([B]) while having a 0 weight 
        """
        self.weights = self.weights.to(x.device)
        return {"cross_entropy_loss": nn.functional.cross_entropy(x.flatten(2, 3), y["label"].flatten(1, 2).long(), weight=self.weights)}

class CrossEntropyIgnore(nn.Module):
    def __init__(self):
        super(CrossEntropyIgnore, self).__init__()

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor BxN that contains the logits
            y: dict that contains "label": torch.Tensor BxN
        Returns:
            torch.Tensor: CrossEntropy loss between x and y: torch.Tensor([B]) while ignoring -1 index
        """
        if len(y["label"].shape) > 1:
            x = x.flatten(2, 3)
            label = y["label"].flatten(1, 2)
        else:
            label = y["label"]
        return {"cross_entropy_loss": nn.functional.cross_entropy(x, label.long(), ignore_index=15)}

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor BxN that contains the logits
            y: dict that contains "label": torch.Tensor BxN
        Returns:
            torch.Tensor: CrossEntropy loss between x and y: torch.Tensor([B])
        """
        return {"cross_entropy_loss": nn.functional.cross_entropy(x, y["label"])}
    

class BCEWithLogs(nn.Module):
    def __init__(self):
        super(BCEWithLogs, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor BxN that contains the logits
            y: dict that contains "label": torch.Tensor BxN
        Returns:
            torch.Tensor: BCE loss between x and y: torch.Tensor([B])
        """
        return {"bce_loss": self.loss(x.float(), y["label"].float())}
    
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor BxN
            y: torch.Tensor BxN
        Returns:
            torch.Tensor: MSE loss between x and y: torch.Tensor([B, N])
        """
        return {"mse_loss": F.smooth_l1_loss(x['predicted_tokens'], y['target'])}

class MAEReconstructionLoss(nn.Module):

    def __init__(self):
        super(MAEReconstructionLoss, self).__init__()

    def forward(self, x , y):
        """
        Args:
            x: dict containing reconstructed tensors of shape BxTxHxWxC
            y: dict containing input tensors of shape BxTxHxWxC
        Returns:
            torch.Tensor: MAE loss
        """
        out, mask = x
        loss = {}
        for i, modality in enumerate(out['kept_modalities']):

            pred = out[f'{modality}_reconstructed'] # [B, T, H, W, C]
            target = y[modality] # [B, T, C, H, W]
            N = out['pos_embed'][:, 1:, :].shape[1]
            H_res = W_res = int(N ** 0.5)
            B, T, C, H_native, W_native = target.shape

            mask_mod = mask[:, i * N : (i + 1) * N]
            mask_mod = mask_mod.view(B, H_res, W_res).unsqueeze(1) # [B, 1, H_res, W_res]
            mask_mod = F.interpolate(mask_mod, size= (H_native, W_native), mode='nearest')
            mask_mod = mask_mod.view(B, 1, H_native*W_native) # [B, 1, H*W]
            mask_mod = mask_mod.expand(-1, T, -1)

            pred = pred.view(B, T, H_native * W_native, C)
            target = target.permute(0, 1, 3, 4, 2).reshape(B, T, H_native * W_native, C)

            mean = target.mean(dim=(-1), keepdim=True)
            var = target.var(dim=(-1), keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

            loss[f'{modality}_reconstruction_loss'] = (pred - target) ** 2
            loss[f'{modality}_reconstruction_loss'] = (loss[f'{modality}_reconstruction_loss']).mean(dim=-1)
            loss[f'{modality}_reconstruction_loss'] = (loss[f'{modality}_reconstruction_loss'] * mask_mod).sum() / mask_mod.sum()

        return loss


LOSSES = {
    "crossentropyweighted": CrossEntropyWeighted,
    "crossentropyignore": CrossEntropyIgnore,
    "crossentropy": CrossEntropy,
    "bce": BCEWithLogs,
    "mse": MSELoss,
    "mae": MAEReconstructionLoss,
}
AVERAGE = {False: lambda x: x, True: lambda x: x.mean(dim=-1)}


class Losses(nn.Module):
    """The Losses meta-object that can take a mix of losses."""

    def __init__(self, mix={}, modalities=[], num_classes=0):
        """Initializes the Losses object.
        Args:
            mix (dict): dictionary with keys "loss_name" and values weight
        """
        super(Losses, self).__init__()
        assert len(mix)
        self.init_losses(mix, modalities, num_classes)

    def init_losses(self, mix, modalities, num_classes):
        """Initializes the losses.
        Args:
            mix (dict): dictionary with keys "loss_name" and values weight
        """
        self.loss = {}
        for m, v in mix.items():
            m = m.lower()
            try:
                if m in ["crossentropyweighted"]:
                    self.loss[m] = (LOSSES[m](num_classes), v)
                elif m in ["milnce"]:
                    self.loss[m] = (LOSSES[m](modalities), v)
                else:
                    self.loss[m] = (LOSSES[m](), v)
            except KeyError:
                raise KeyError(f"Loss {m} not found in {LOSSES.keys()}")

    def forward(self, x, y, average=True):
        """Computes the losses.
        Args:
            x: dict that contains "gps": torch.Tensor Bx2 or "label": torch.Tensor BxN
            y: dict that contains "gps": torch.Tensor Bx2 or "label": torch.Tensor BxN
            average (bool): whether to average the losses or not
        Returns:
            dict: dictionary with losses
        """
        output = {"loss": 0}
        for loss_name, (loss, weight) in self.loss.items():
            loss_output = loss(x, y)
            for k, v in loss_output.items():
                if k.endswith("_loss"):
                    v = AVERAGE[average](v)
                    output["loss"] += weight * v
                output[k] = v
        return output
