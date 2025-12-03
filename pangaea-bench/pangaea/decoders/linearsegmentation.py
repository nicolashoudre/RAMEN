import torch
import torch.nn as nn
import torch.nn.functional as F

from pangaea.decoders.base import Decoder
from pangaea.encoders.base import Encoder


class LinearSegmentation(Decoder):

    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
    ):
        """ Linear decoder for segmentation tasks. 

        Args:
            encoder (Encoder): Model used for feature extraction. Must define a forward(images) method
                that returns a feature tensor.
            num_classes (int): number of classes in the dataset.
            finetune (bool): whether to finetune the encoder.
            feature_multiplier (int, optional): feature multiplier. Defaults to 1.
            in_channels (list[int], optional): input channels. Defaults to None.
        """
        
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
        )

        self.model_name = "LinearSegmentation"
        self.encoder = encoder
        self.finetune = finetune

        if not self.finetune:
            for param in self.encoder.parameters():
                param.requires_grad = False
        

        self.num_classes = num_classes
        self.linear_head = nn.Conv2d(self.encoder.output_dim[0], num_classes, kernel_size=1)
    
    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        """Compute the segmentation output.

        Args:
            img (dict[str, torch.Tensor]): input data structured as a dictionary:
            img = {modality1: tensor1, modality2: tensor2, ...}, e.g. img = {"optical": tensor1, "sar": tensor2}.
            with tensor1 and tensor2 of shape (B C T=1 H W) with C the number of encoders'bands for the given modality.
            output_shape (torch.Size | None, optional): output's spatial dims (H, W) (equals to the target spatial dims).
            Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape (B, num_classes, H', W') with (H' W') coressponding to the output_shape.
        """

        # img[modality] of shape [B C T>1 H W]
        if self.encoder.multi_temporal:
            if not self.finetune:
                with torch.no_grad():
                    feat = self.encoder(img)
            else:
                feat = self.encoder(img)

            # multi_temporal models can return either (B C' T=1 H' W')
            # or (B C' H' W'), we need (B C' H' W')
            if self.encoder.multi_temporal_output:
                feat = [f.squeeze(-3) for f in feat]

        else:
            # remove the temporal dim
            # [B C T=1 H W] -> [B C H W]
            if not self.finetune:
                with torch.no_grad():
                    feat = self.encoder({k: v[:, :, 0, :, :] for k, v in img.items()})
            else:
                feat = self.encoder({k: v[:, :, 0, :, :] for k, v in img.items()})
        
        output = self.linear_head(feat[-1])

        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]

        # interpolate to the target spatial dims
        output = F.interpolate(output, size=output_shape, mode="bilinear")

        return output

class LinearMTSegmentation(Decoder):

    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
        multi_temporal: int,
    ):
        """ Linear decoder for segmentation tasks. 

        Args:
            encoder (Encoder): Model used for feature extraction. Must define a forward(images) method
                that returns a feature tensor.
            num_classes (int): number of classes in the dataset.
            finetune (bool): whether to finetune the encoder.
            feature_multiplier (int, optional): feature multiplier. Defaults to 1.
            in_channels (list[int], optional): input channels. Defaults to None.
        """
        
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
        )

        self.model_name = "LinearSegmentation"
        self.encoder = encoder
        self.finetune = finetune
        self.multi_temporal = multi_temporal

        if not self.finetune:
            for param in self.encoder.parameters():
                param.requires_grad = False
        

        self.num_classes = num_classes
        self.tmap = nn.Linear(self.multi_temporal, 1)
        self.linear_head = nn.Conv2d(self.encoder.output_dim[0], num_classes, kernel_size=1)
    
    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        """Compute the segmentation output.

        Args:
            img (dict[str, torch.Tensor]): input data structured as a dictionary:
            img = {modality1: tensor1, modality2: tensor2, ...}, e.g. img = {"optical": tensor1, "sar": tensor2}.
            with tensor1 and tensor2 of shape (B C T=1 H W) with C the number of encoders'bands for the given modality.
            output_shape (torch.Size | None, optional): output's spatial dims (H, W) (equals to the target spatial dims).
            Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape (B, num_classes, H', W') with (H' W') coressponding to the output_shape.
        """

        # img[modality] of shape [B C T>1 H W]
        if self.encoder.multi_temporal:
            if not self.finetune:
                with torch.no_grad():
                    feat = self.encoder(img)
            else:
                feat = self.encoder(img)

            # multi_temporal models can return either (B C' T=1 H' W')
            # or (B C' H' W'), we need (B C' H' W')
            if self.encoder.multi_temporal_output:
                feat = [f.squeeze(-3) for f in feat]

        else:
            feats = []
            for i in range(self.multi_temporal):
                inputs = {k: v[:, :, i, :, :] for k, v in img.items() if not k.endswith("_dates")}
            
                if not self.finetune:
                    with torch.no_grad():
                        feats.append(self.encoder(inputs))
                else:
                    feats.append(self.encoder(inputs))

            feats = [list(i) for i in zip(*feats)]
            # obtain features per layer
            feats = [torch.stack(feat_layers, dim=2) for feat_layers in feats]
            feat = [self.tmap(f.permute(0, 1, 3, 4, 2)).squeeze(-1) for f in feats]
        
        output = self.linear_head(feat[-1])

        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]

        # interpolate to the target spatial dims
        output = F.interpolate(output, size=output_shape, mode="bilinear")

        return output
