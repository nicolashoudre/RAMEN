import torch
from torch import nn
from torch.nn import functional as F
from models.networks.encoder.utils.ramen_utils import ScaleResampler, SpectralProjector, RadarProjector, DemProjector
from models.networks.encoder.utils.ltae import LTAE2d
from models.networks.encoder.utils.pos_embed import get_2d_sincos_pos_embed_with_resolution
from timm.models.vision_transformer import Block
from timm.layers.attention_pool import AttentionPoolLatent
from models.networks.encoder.utils.utils import trunc_normal_
import random
from einops import rearrange
from collections import defaultdict

class RamenViT(nn.Module):
    """
    Initialize RAMEN encoder with multiple resolutions per modality and multiple datasets handling
    Args:
        datasets (list): List of datasets to be processed
        modalities (dict): dict with available modalities for each dataset
        modalities_chans (dict): nested dict mapping datasets to their modalities and channel sizes
        modalities_res (dict): nested dict mapping datasets to their modalities and resolutions
        modalities_input_size (dict): nested dict mapping datasets to their modalities and input sizes
        num_iterations (int): Number of iterations for resampling
        all_res (dict): dict mapping datasets to their all resolutions to be used
    """
    def __init__(
        self,
        modalities: list = [],
        modalities_res: dict = {},
        modalities_input_size: dict = {},
        num_iterations: int = 1,
        all_res: list = [],
        wavelengths: dict = {},
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qvk_bias: bool = True,
        qk_scale: float = None,
        class_token: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        
        super().__init__()

        self.modalities = modalities
        self.modalities_res = modalities_res
        self.modalities_input_size = modalities_input_size
        self.all_res = all_res
        self.wavelengths = wavelengths
        self.embed_dim = embed_dim

        self.num_iterations = num_iterations

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None

        self.spectral_projector = SpectralProjector(embed_dim, embed_dim*2)
        if "s1" in self.modalities or "alos" in self.modalities or "s1_des" in self.modalities:
            self.radar_projector = RadarProjector(embed_dim, embed_dim*2)
        if "dem" in self.modalities:
            self.dem_projector = DemProjector(embed_dim, embed_dim*2)

        if self.num_iterations > 0:
            self.resampler = ScaleResampler(embed_dim)

        self.ltae = LTAE2d(
                    in_channels=embed_dim,
                    n_head=16,
                    d_k=16,
                    mlp= [embed_dim, embed_dim*2, embed_dim],
                    dropout=0.,
                    mlp_in= [embed_dim],
                    T= 367,
                    in_norm=True,
                    return_att=False,
                    positional_encoding=True,
                )        

        self.in_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qvk_bias,
                qk_norm=qk_scale,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

        self.pos_embed_dict = defaultdict(dict)
        self.effective_size_dict = defaultdict(dict)
        for res in self.all_res:
            effective_size = int(modalities_input_size[modalities[0]] * (modalities_res[modalities[0]] / res))
            self.effective_size_dict[res] = effective_size
            self.pos_embed_dict[res] = get_2d_sincos_pos_embed_with_resolution(
                embed_dim,
                effective_size,
                torch.tensor([res]),
                cls_token=True,
            )
    
    def forward_proj(self, x, effective_res, modalities_keep=None):
        """
        Forward pass the Projection + Resampler
        Apply target gsd to each modality
        effective_res: dict with a resolution for each modality
        """
        out = {}
        out['kept_modalities'] = modalities_keep if modalities_keep is not None else self.modalities
        out['effective_res'] = effective_res

        device = x[out['kept_modalities'][0]].device
        dtype = x[out['kept_modalities'][0]].dtype
        out['pos_embed'] = self.pos_embed_dict[effective_res].to(device=device, dtype=dtype)
        
        for modality in out['kept_modalities']:
            x_mod = x[modality]   # [B, T, C, H, W]
            B, T, C, H, W = x_mod.shape
            x_mod = x_mod.permute(0, 1, 3, 4, 2).contiguous()  # [B, T, H, W, C]
        
            if modality in ["s1", "alos", "s1_des", "s1_asc"]:
                spectral_encoding = self.radar_projector(
                    self.wavelengths[modality], device
                )
            elif modality in ["dem"]:
                spectral_encoding = self.dem_projector(
                    self.wavelengths[modality], device
                )
            else:
                spectral_encoding = self.spectral_projector(
                    torch.Tensor(self.wavelengths[modality]).to(device=device, dtype=dtype)
                )
        
            out_mod = x_mod @ spectral_encoding  # [B, T, H, W, C] @ [C, D] -> [B, T, H, W, D]
            out_mod = out_mod.permute(0, 1, 4, 2, 3).reshape(B * T, -1, H, W).contiguous()  # [B*T, D, H, W]
        
            expected_size = self.effective_size_dict[effective_res]
        
            if self.num_iterations > 0:
                scale = (
                    self.modalities_res[modality] / effective_res
                ) ** (1 / self.num_iterations)
                for _ in range(self.num_iterations):
                    out_mod = self.resampler(out_mod, scale)
        
            if out_mod.shape[2] != expected_size or out_mod.shape[3] != expected_size:
                out_mod = F.interpolate(out_mod, size=(expected_size, expected_size), mode='bilinear')

            if T > 1:
                out_mod = out_mod.view(B, T, -1, expected_size, expected_size)  # [B, T, D, H, W]
                out_mod = self.ltae(
                    out_mod, batch_positions=x['_'.join([modality, "dates"])], pad_mask=None
                )  # [B, D, H, W]
            else:
                out_mod.unsqueeze(1)
            out_mod = out_mod.flatten(2).transpose(1, 2)  # [B, (H*W), D]
            out_mod = self.in_norm(out_mod)
        
            out[modality] = out_mod
            out[f'{modality}_TS'] = T
            #out[f'{modality}_dates'] = x['_'.join([modality, "dates"])]

        tokens = torch.cat([out[modality] for modality in out['kept_modalities']], dim=1)
        tokens = tokens + out['pos_embed'][:, 1:, :].repeat(1, len(out['kept_modalities']), 1)
        out['tokens'] = tokens

        return out

    def forward_transformer(self, out):
        """
        Forward pass of the Transformer blocks
        """

        if self.cls_token is not None:
            cls_tokens = (self.cls_token + out['pos_embed'][:, :1, :]).expand(out['tokens'].shape[0], -1, -1)
            tokens = torch.cat((cls_tokens, out['tokens']), dim=1)
        else:
            tokens = out['tokens']

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        return tokens

    def forward(self, x, effective_res, modalities_keep=None):
        """
        Forward pass of the network
        """
        out = {}
        out['kept_modalities'] = modalities_keep if modalities_keep is not None else self.modalities
        out['effective_res'] = effective_res

        device = x[out['kept_modalities'][0]].device
        dtype = x[out['kept_modalities'][0]].dtype
        out['pos_embed'] = self.pos_embed_dict[effective_res].to(device=device, dtype=dtype)

        for modality in out['kept_modalities']:
            x_mod = x[modality] # [B, T, C, H, W]
            B, T, C, H, W = x_mod.shape
            x_mod = x_mod.permute(0, 1, 3, 4, 2).contiguous()

            if modality in ["s1", "alos", "s1_des", "s1_asc"]:
                spectral_encoding = self.radar_projector(
                    self.wavelengths[modality], device
                )
            elif modality in ["dem"]:
                spectral_encoding = self.dem_projector(
                    self.wavelengths[modality], device
                )
            else:
                spectral_encoding = self.spectral_projector(
                    torch.Tensor(self.wavelengths[modality]).to(device=device, dtype=dtype)
                )

            out_mod = x_mod @ spectral_encoding # [B, T, H, W, C] @ [C, D] -> [B, T, H, W, D]
            out_mod = out_mod.permute(0, 1, 4, 2, 3).reshape(B * T, -1, H, W).contiguous() # [B*T, D, H, W]

            expected_size = self.effective_size_dict[effective_res]
            
            if self.num_iterations > 0:
                scale = (
                    self.modalities_res[modality] / effective_res
                ) ** (1 / self.num_iterations)
                for _ in range(self.num_iterations):
                    out_mod = self.resampler(out_mod, scale)
                    
            if out_mod.shape[2] != expected_size or out_mod.shape[3] != expected_size:
                out_mod = F.interpolate(out_mod, size=(expected_size, expected_size), mode='bilinear')

            if T > 1:
                out_mod = out_mod.view(B, T, -1, expected_size, expected_size)  # [B, T, D, H, W]
                out_mod = self.ltae(
                    out_mod, batch_positions=x['_'.join([modality, "dates"])], pad_mask=None
                )  # [B, D, H, W]
            else:
                out_mod.unsqueeze(1)
            out_mod = out_mod.flatten(2).transpose(1, 2)  # [B, (H*W), D]
            out_mod = self.in_norm(out_mod)

            out[modality] = out_mod
            out[f'{modality}_TS'] = T
            #out[f'{modality}_dates'] = x['_'.join([modality, "dates"])]

        tokens = torch.cat([out[modality] for modality in out['kept_modalities']], dim=1)
        tokens = tokens + out['pos_embed'][:, 1:, :].repeat(1, len(out['kept_modalities']), 1)

        if self.cls_token is not None:
            cls_tokens = (self.cls_token + out['pos_embed'][:, :1, :]).expand(tokens.shape[0], -1, -1)
            tokens = torch.cat((cls_tokens, tokens), dim=1)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        out['tokens'] = tokens

        return out

class RamenViTMultiData(nn.Module):
    """
    Initialize RAMEN encoder with multiple resolutions per modality and multiple datasets handling
    Args:
        datasets (list): List of datasets to be processed
        modalities (dict): dict with available modalities for each dataset
        modalities_chans (dict): nested dict mapping datasets to their modalities and channel sizes
        modalities_res (dict): nested dict mapping datasets to their modalities and resolutions
        modalities_input_size (dict): nested dict mapping datasets to their modalities and input sizes
        num_iterations (int): Number of iterations for resampling
        all_res (dict): dict mapping datasets to their all resolutions to be used
    """
    def __init__(
        self,
        datasets: list = [],
        modalities: dict = {},
        modalities_res: dict = {},
        modalities_input_size: dict = {},
        num_iterations: int = 1,
        all_res: dict = {},
        wavelengths: dict = {},
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qvk_bias: bool = True,
        qk_scale: float = None,
        class_token: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        
        super().__init__()

        self.modalities = modalities
        self.modalities_res = modalities_res
        self.modalities_input_size = modalities_input_size
        self.all_res = all_res
        self.wavelengths = wavelengths
        self.embed_dim = embed_dim

        self.num_iterations = num_iterations

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None

        self.spectral_projector = SpectralProjector(embed_dim, embed_dim*2)
        if any(value in ["s1_des", "s1", "alos"] for sublist in self.modalities.values() for value in sublist):
            self.radar_projector = RadarProjector(embed_dim, embed_dim*2)
        if any(value in ["dem"] for sublist in self.modalities.values() for value in sublist):
            self.dem_projector = DemProjector(embed_dim, embed_dim*2)

        if self.num_iterations > 0:
            self.resampler = ScaleResampler(embed_dim)

        self.ltae = LTAE2d(
                    in_channels=embed_dim,
                    n_head=16,
                    d_k=16,
                    mlp= [embed_dim, embed_dim*2, embed_dim],
                    dropout=0.,
                    mlp_in= [embed_dim],
                    T= 367,
                    in_norm=True,
                    return_att=False,
                    positional_encoding=True,
                )        

        self.in_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qvk_bias,
                qk_norm=qk_scale,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

        self.pos_embed_dict = defaultdict(dict)
        self.effective_size_dict = defaultdict(dict)
        for dataset in datasets:
            for res in self.all_res[dataset]:
                effective_size = int(modalities_input_size[dataset][modalities[dataset][0]] * (modalities_res[dataset][modalities[dataset][0]] / res))
                self.effective_size_dict[dataset][res] = effective_size
                self.pos_embed_dict[dataset][res] = get_2d_sincos_pos_embed_with_resolution(
                    embed_dim,
                    effective_size,
                    torch.tensor([res]),
                    cls_token=True,
                )
    
    def forward_proj(self, x, effective_res, modalities_keep=None):
        """
        Forward pass the Projection + Resampler
        Apply target gsd to each modality
        effective_res: dict with a resolution for each modality
        """
        out = {}
        out['kept_modalities'] = modalities_keep if modalities_keep is not None else self.modalities
        out['dataset'] = x['dataset']
        out['effective_res'] = effective_res

        device = x[out['kept_modalities'][0]].device
        dtype = x[out['kept_modalities'][0]].dtype
        out['pos_embed'] = self.pos_embed_dict[out['dataset']][effective_res].to(device=device, dtype=dtype)
        
        for modality in out['kept_modalities']:
            x_mod = x[modality]   # [B, T, C, H, W]
            B, T, C, H, W = x_mod.shape
            x_mod = x_mod.permute(0, 1, 3, 4, 2).contiguous()  # [B, T, H, W, C]
        
            if modality in ["s1", "alos", "s1_des", "s1_asc"]:
                spectral_encoding = self.radar_projector(
                    self.wavelengths[out['dataset']][modality], device
                )
            elif modality in ["dem"]:
                spectral_encoding = self.dem_projector(
                    self.wavelengths[out['dataset']][modality], device
                )
            else:
                spectral_encoding = self.spectral_projector(
                    torch.Tensor(self.wavelengths[out['dataset']][modality]).to(device=device, dtype=dtype)
                )
        
            out_mod = x_mod @ spectral_encoding  # [B, T, H, W, C] @ [C, D] -> [B, T, H, W, D]
            out_mod = out_mod.permute(0, 1, 4, 2, 3).reshape(B * T, -1, H, W)  # [B*T, D, H, W]
        
            expected_size = self.effective_size_dict[out['dataset']][effective_res]
        
            if self.num_iterations > 0:
                scale = (
                    self.modalities_res[out['dataset']][modality] / effective_res
                ) ** (1 / self.num_iterations)
                for _ in range(self.num_iterations):
                    out_mod = self.resampler(out_mod, scale)
        
            if out_mod.shape[2] != expected_size or out_mod.shape[3] != expected_size:
                out_mod = F.interpolate(out_mod, size=(expected_size, expected_size), mode='bilinear')
        
            if T > 1:
                out_mod = out_mod.view(B, T, -1, expected_size, expected_size)  # [B, T, D, H, W]
                out_mod = self.ltae(
                    out_mod, batch_positions=x['_'.join([modality, "dates"])], pad_mask=None
                )  # [B, D, H, W]
            else:
                out_mod.unsqueeze(1)
            out_mod = out_mod.flatten(2).transpose(1, 2)  # [B, (H*W), D]
            out_mod = self.in_norm(out_mod)
        
            out[modality] = out_mod
            out[f'{modality}_TS'] = T
            out[f'{modality}_dates'] = x['_'.join([modality, "dates"])]

        tokens = torch.cat([out[modality] for modality in out['kept_modalities']], dim=1)
        tokens = tokens + out['pos_embed'][:, 1:, :].repeat(1, len(out['kept_modalities']), 1)
        out['tokens'] = tokens

        return out

    def forward_transformer(self, out):
        """
        Forward pass of the Transformer blocks
        """

        if self.cls_token is not None:
            cls_tokens = (self.cls_token + out['pos_embed'][:, :1, :]).expand(out['tokens'].shape[0], -1, -1)
            tokens = torch.cat((cls_tokens, out['tokens']), dim=1)
        else:
            tokens = out['tokens']

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        return tokens

    def forward(self, x, effective_res, modalities_keep=None):
        """
        Forward pass of the network
        """
        out = {}
        out['kept_modalities'] = modalities_keep if modalities_keep is not None else self.modalities
        out['dataset'] = x['dataset']
        out['effective_res'] = effective_res

        device = x[out['kept_modalities'][0]].device
        dtype = x[out['kept_modalities'][0]].dtype
        out['pos_embed'] = self.pos_embed_dict[out['dataset']][effective_res].to(device=device, dtype=dtype)

        for modality in out['kept_modalities']:
            x_mod = x[modality] # [B, T, C, H, W]
            B, T, C, H, W = x_mod.shape
            x_mod = x_mod.permute(0, 1, 3, 4, 2).contiguous()

            if modality in ["s1", "alos", "s1_des", "s1_asc"]:
                spectral_encoding = self.radar_projector(
                    self.wavelengths[out['dataset']][modality], device
                )
            elif modality in ["dem"]:
                spectral_encoding = self.dem_projector(
                    self.wavelengths[out['dataset']][modality], device
                )
            else:
                spectral_encoding = self.spectral_projector(
                    torch.Tensor(self.wavelengths[out['dataset']][modality]).to(device=device, dtype=dtype)
                )

            out_mod = x_mod @ spectral_encoding # [B, T, H, W, C] @ [C, D] -> [B, T, H, W, D]
            out_mod = out_mod.permute(0, 1, 4, 2, 3).reshape(B * T, -1, H, W) # [B*T, D, H, W]

            expected_size = self.effective_size_dict[out['dataset']][effective_res]
            
            if self.num_iterations > 0:
                scale = (
                    self.modalities_res[out['dataset']][modality] / effective_res
                ) ** (1 / self.num_iterations)
                for _ in range(self.num_iterations):
                    out_mod = self.resampler(out_mod, scale)
                    
            if out_mod.shape[2] != expected_size or out_mod.shape[3] != expected_size:
                out_mod = F.interpolate(out_mod, size=(expected_size, expected_size), mode='bilinear')

            if T > 1:
                out_mod = out_mod.view(B, T, -1, expected_size, expected_size)  # [B, T, D, H, W]
                out_mod = self.ltae(
                    out_mod, batch_positions=x['_'.join([modality, "dates"])], pad_mask=None
                )  # [B, D, H, W]
            else:
                out_mod.unsqueeze(1)
            out_mod = out_mod.flatten(2).transpose(1, 2)  # [B, (H*W), D]
            out_mod = self.in_norm(out_mod)

            out[modality] = out_mod
            out[f'{modality}_TS'] = T
            out[f'{modality}_dates'] = x['_'.join([modality, "dates"])]

        tokens = torch.cat([out[modality] for modality in out['kept_modalities']], dim=1)
        tokens = tokens + out['pos_embed'][:, 1:, :].repeat(1, len(out['kept_modalities']), 1)

        if self.cls_token is not None:
            cls_tokens = (self.cls_token + out['pos_embed'][:, :1, :]).expand(tokens.shape[0], -1, -1)
            tokens = torch.cat((cls_tokens, tokens), dim=1)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        out['tokens'] = tokens

        return out



