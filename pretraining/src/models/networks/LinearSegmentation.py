import torch
import torch.nn as nn
from torch.nn import functional as F
from models.networks.encoder.utils.vit_utils import AttentionPoolLatent
from einops import rearrange
import functools

class LinearSegmentation(nn.Module):
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
        self.res = self.encoder.all_res # To be ordered
        
        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        self.num_classes = num_classes
        self.use_attn_pooling=use_attn_pooling
        
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

        self.conv_seg = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=1)

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

        feat = feat[0]
        output = self.conv_seg(feat)

        output = F.interpolate(output, size=self.pred_size, mode="bilinear")
            
        return output
