import torch
import torch.nn as nn
from torch.nn import functional as F
from models.networks.encoder.utils.vit_utils import AttentionPoolLatent
from einops import rearrange
import functools

class RaPerNet(nn.Module):
    """
    Initialize Fine Tuning of OmniSat after pretraining
    Args:
        encoder (torch.nn.Module): initialized model
        path (str): path of checkpoint of model to load
        name (str): name of the weights from checkpoint to use
        freeze (bool); if True, freeze encoder to perform linear probing
        n_class (int): output_size of mlp
        modalities (list): list of modalities to use
    """
    def __init__(self, 
                 encoder: torch.nn.Module,
                 path: str = '',
                 pred_size: int = 512,
                 name: str = 'encoder',
                 freeze: bool = True,
                 num_classes: int = 15,
                 modalities: list = [],
                 use_attn_pooling: bool = False,
                 channels: int = 512,
                 pool_scales=(1, 2, 3, 6),
                ):
        super().__init__()

        self.pred_size = pred_size
        self.freeze = freeze
        self.modalities = modalities

        target = (name == "encoder")
        
        if path is not None:
            u = torch.load(path, map_location="cpu")
            d = {}
            for key in u["state_dict"].keys():
                if name in key:
                    if target:
                        if 'pixel_encoders' in key:
                            if any([modality in key for modality in modalities]):
                                clean_key = key
                                for prefix in ["model.encoder.", "encoder."]:
                                    if key.startswith(prefix):
                                        clean_key = key[len(prefix):]
                                        break
                                d[clean_key] = u["state_dict"][key]
                        else:
                            if not('predictor.' in key):
                                clean_key = key
                                for prefix in ["model.encoder.", "encoder."]:
                                    if key.startswith(prefix):
                                        clean_key = key[len(prefix):]
                                        break
                                d[clean_key] = u["state_dict"][key]
        
            load_result = encoder.load_state_dict(d, strict=False)
        
            print("Missing keys (not loaded):")
            print(load_result.missing_keys)
        
            print("Unexpected keys (in checkpoint but not in model):")
            print(load_result.unexpected_keys)
        
            print(f"Loaded {len(d)} keys into encoder.")
        
            del u
            del d
                 
        self.encoder = encoder
        self.embed_dim = self.encoder.embed_dim
        self.in_channels = [self.embed_dim, self.embed_dim, self.embed_dim, self.embed_dim]
        self.res = self.encoder.all_res # To be ordered
        
        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        self.num_classes = num_classes
        self.use_attn_pooling = use_attn_pooling

        if use_attn_pooling is True:
            self.modality_head = nn.ModuleList([
                AttentionPoolLatent(in_features=self.embed_dim, num_heads=4) 
                for _ in range(len(self.res))
            ])
        else:
            self.modality_head = nn.ModuleList([
                nn.Linear(len(self.modalities)*self.embed_dim, self.embed_dim)
                    for _ in range(len(self.res))
            ])

        scales = [4, 2, 1, 0.5]
        rescales = [
            scales[int(i / self.input_layers_num * 4)]
            for i in range(self.input_layers_num)
        ]

        self.neck = Feature2Pyramid(
            embed_dim=self.in_channels,
            rescales=rescales,
        )

        self.channels = channels
        self.align_corners = False

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels[-1] + len(pool_scales) * self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SyncBatchNorm(self.channels),
            nn.ReLU(inplace=True),
        )

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    padding=0,
                ),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=False),
            )
            fpn_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.SyncBatchNorm(self.channels),
                nn.ReLU(inplace=False),
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=len(self.in_channels) * self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SyncBatchNorm(self.channels),
            nn.ReLU(inplace=True),
        )

        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)
        
    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, x):
        """
        Forward pass of the network.
        """
        
        M = len(self.modalities)
        feat=[]
        for i,r in enumerate(self.res):
            x_r = self.encoder(x, r)
            x_r = x_r['tokens'][:, 1:, :]
            B = x_r.shape[0]
            N = x_r.shape[1]
            P = N // M
            H = W = int(P ** 0.5)
            x_r = rearrange(x_r, 'b (m h w) d -> b h w m d', m=M, h=H, w=W)
            if self.use_attn_pooling is True:
                x_r = rearrange(x_r, 'b h w m d -> (b h w) m d')
                x_r = self.modality_head[i](x_r)  
                x_r = rearrange(x_r, '(b h w) d -> b h w d', b=B, h=H, w=W)
            else:  
                x_r = rearrange(x_r, 'b h w m d -> b h w (m d)')
                x_r = self.modality_head[i](x_r)  # [B, H, W, D]
            x_r = rearrange(x_r, 'b h w d -> b d h w')
            feat.append(x_r)
            
        feat = self._forward_feature(feat)
        feat = self.dropout(feat)
        output = self.conv_seg(feat)

        output = F.interpolate(output, size=self.pred_size, mode="bilinear")
            
        return output


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners, **kwargs):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=self.channels,
                        kernel_size=1,
                        padding=0,
                    ),
                    nn.SyncBatchNorm(self.channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


if __name__ == "__main__":
    _ = Fine()
