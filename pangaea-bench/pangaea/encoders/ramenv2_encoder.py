# RAMEN Encoder 

from functools import partial
from logging import Logger
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from timm.models.vision_transformer import Block
from collections import defaultdict

from einops import rearrange

import copy
import numpy as np

from pangaea.encoders.base import Encoder
from pangaea.encoders.pos_embed import get_1d_sincos_pos_embed_from_grid_torch, get_2d_sincos_pos_embed_with_resolution

class RAMENv2_Encoder(Encoder):
    """
    Paper: https://arxiv.org/pdf/2403.15356
    Attributes:
        output_layers (int | list[int]): The layers from which to extract the output.
        img_size (int): The size of the input image.
        wv_planes (int): The number of wavelet planes.
        wave_list (dict[str, dict[str, float]]): A dictionary containing wavelet information for each band.
        return_all_tokens (bool): Whether to return all tokens or not.
        embed_dim (int): The embedding dimension.
        use_norm (bool): Whether to use normalization or not.
        wv_list (list[float]): A list of wavelet values for each band.
        norm (nn.Module): The normalization layer.
        patch_embed (Dynamic_MLP_OFA): The patch embedding layer.
        num_patches (int): The number of patches in the input image.
        cls_token (nn.Parameter): The class token parameter.
        pos_embed (nn.Parameter): The positional embedding parameter.
        blocks (nn.ModuleList): A list of Transformer blocks.
    Methods:
        __init__(encoder_weights, input_bands, input_size, embed_dim, output_layers, wave_list, patch_size=16, depth=12, num_heads=16, wv_planes=128, return_all_tokens=True, mlp_ratio=4., use_norm=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
            Initializes the RAMEN_Encoder with the given parameters.
        forward(image):
            Forward pass of the encoder. Takes an input image and returns the encoded output.
        load_encoder_weights(logger):
            Loads the encoder weights from a pretrained model and logs any missing or incompatible parameters.
    """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_bands: dict[str, list[str]],
        input_size: int,
        embed_dim: int,
        output_dim: int | list[int],
        output_layers: int | list[int],
        download_url: str,
        modalities: list = [],
        modalities_res: dict = {},
        modalities_input_size: dict = {},
        all_res: list = [],
        wavelengths: list = [],
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        class_token: bool = True,
        qvk_bias: bool = True,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__(
            model_name="ramen_encoder",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=download_url,
        )

        self.output_layers = output_layers
        self.img_size = input_size

        self.modalities = modalities
        self.modalities_res = modalities_res
        self.modalities_input_size = modalities_input_size
        self.all_res = all_res
        self.wavelengths = wavelengths
        self.embed_dim = embed_dim


        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None

        self.spectral_projector = SpectralProjector(embed_dim, embed_dim*2)
        if "s1" in self.modalities or "alos" in self.modalities or "s1_des" in self.modalities or "sar" in self.modalities:
            self.radar_projector = RadarProjector(embed_dim, embed_dim*2)
        if "dem" in self.modalities:
            self.dem_projector = DemProjector(embed_dim, embed_dim*2)

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

        #if self.cls_token is not None:
            #trunc_normal_(self.cls_token, std=.02)

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

    def forward(self, x):
        device = x[self.modalities[0]].device
        dtype = x[self.modalities[0]].dtype
        output = []
        for i, r in enumerate(self.all_res):
            out = {}
            out['pos_embed'] = self.pos_embed_dict[r].to(device=device, dtype=dtype)
            for modality in self.modalities:
                x_mod = x[modality] # [B, C, T, H, W]
                B, C, H, W = x_mod.shape
                x_mod = x_mod.permute(0, 2, 3, 1).contiguous() # [B, H, W, C]

                if modality in ["s1", "alos", "s1_des", "s1_asc", "sar"]:
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
                #spectral_encoding = spectral_encoding.T.contiguous().view(self.embed_dim, C, 1, 1)
                #out_mod = F.conv2d(x_mod, spectral_encoding)
                out_mod = x_mod @ spectral_encoding # [B, H, W, C] @ [C, D] -> [B, H, W, D]
                out_mod = out_mod.permute(0, 3, 1, 2).contiguous()  #.reshape(B, -1, H, W).contiguous() # [B, D, H, W]

                expected_size = self.effective_size_dict[r]
                scale = (self.modalities_res[modality] / r)
                out_mod = self.resampler(out_mod, scale)
                    
                if out_mod.shape[2] != expected_size or out_mod.shape[3] != expected_size:
                    out_mod = F.interpolate(out_mod, size=(expected_size, expected_size), mode='bilinear')

                out_mod = out_mod.flatten(2).transpose(1, 2)  # [B, (H*W), D]
                out_mod = self.in_norm(out_mod)
                out[modality] = out_mod

            tokens = torch.cat([out[modality] for modality in self.modalities], dim=1)
            tokens = tokens + out['pos_embed'][:, 1:, :].repeat(1, len(self.modalities), 1)

            if self.cls_token is not None:
                cls_tokens = (self.cls_token + out['pos_embed'][:, :1, :]).expand(tokens.shape[0], -1, -1)
                tokens = torch.cat((cls_tokens, tokens), dim=1)

            for j, blk in enumerate(self.blocks):
                tokens = blk(tokens)
                if j == len(self.blocks) - 1:
                    tokens = self.norm(tokens)
                if j in self.output_layers:
                    if len(self.modalities) > 1:
                        n_tok = tokens[:, 1:, :].shape[1] // len(self.modalities)
                        patch_tokens = [tokens[:, 1 + i*n_tok : 1 + (i+1)*n_tok, :] for i in range(len(self.modalities))]
                        patch_tokens = torch.cat(patch_tokens, dim=-1)
                        out = (
                            patch_tokens
                            .permute(0, 2, 1)
                            .view(
                                patch_tokens.shape[0],
                                -1,
                                expected_size,
                                expected_size,
                            )
                            .contiguous()
                        )
                    else:
                        out = (
                            tokens[:, 1:]
                            .permute(0, 2, 1)
                            .view(
                                tokens.shape[0],
                                -1,
                                expected_size,
                                expected_size,
                            )
                            .contiguous()
                        )
                    output.append(out)

        return output

    def load_encoder_weights(self, logger: Logger) -> None:
        
        pass

class SpectralEncoding(nn.Module):

    def __init__(self, embed_dim: int, normalize: bool = False):
        """
        Args:
            embed_dim: Dimension of the positional encoding.
            normalize: Whether to normalize wavelengths before encoding.
        """
        super().__init__()
        assert embed_dim % 2 == 0, "Embedding dimension must be even"
        self.embed_dim = embed_dim
        self.normalize = normalize

    def forward(self, wavelengths: Tensor) -> Tensor:
        """
        Args:
            wavelengths: Tensor of spectral positions [C].
                        e.g., wavelengths of each input channel.
        Returns:
            pos_emb: [C, embed_dim]
        """
        if self.normalize:
            wavelengths = (wavelengths - 350.0) / (2500.0 - 350.0)
            wavelengths = wavelengths.clamp(0, 1)

        pos_emb = get_1d_sincos_pos_embed_from_grid_torch(self.embed_dim, wavelengths)
        return pos_emb

class SpectralProjector(nn.Module):

    def __init__(self, embed_dim=768, hidden_dim=768):

        super().__init__()
        self.embed_dim = embed_dim

        self.spectral_encoding = SpectralEncoding(embed_dim, normalize=False)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, wavelengths: Tensor) -> Tensor:
        """
        Args:
            wavelengths: Tensor of spectral positions [C].
                        e.g., wavelengths of each input channel.
        Returns:
            pos_emb: [C, embed_dim]
        """
        spectral_encodings = self.spectral_encoding(wavelengths)
        return self.mlp(spectral_encodings)

class RadarProjector(nn.Module):

    def __init__(self, embed_dim=768, hidden_dim=768):

        super().__init__()
        self.embed_dim = embed_dim

        self.asc_vv_param = nn.Parameter(torch.empty(embed_dim))
        self.asc_vh_param = nn.Parameter(torch.empty(embed_dim))
        self.asc_hv_param = nn.Parameter(torch.empty(embed_dim))
        self.asc_hh_param = nn.Parameter(torch.empty(embed_dim))
        
        self.des_vv_param = nn.Parameter(torch.empty(embed_dim))
        self.des_vh_param = nn.Parameter(torch.empty(embed_dim))
        self.des_hv_param = nn.Parameter(torch.empty(embed_dim))
        self.des_hh_param = nn.Parameter(torch.empty(embed_dim))

        for name, param in self.named_parameters():
            if "param" in name:
                nn.init.normal_(param, 0.0, 0.02)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.pol_map = {
            "asc_vv": self.asc_vv_param,
            "asc_vh": self.asc_vh_param,
            "asc_hv": self.asc_hv_param,
            "asc_hh": self.asc_hh_param,
            "des_vv": self.des_vv_param,
            "des_vh": self.des_vh_param,
            "des_hv": self.des_hv_param,
            "des_hh": self.des_hh_param,
        }

    def forward(self, polarizations: list, device=None) -> torch.Tensor:
        """
        Args:
            polarizations: List of polarization strings ['vv', 'vh', 'hh', 'hv'] of length [C].
            device: torch.device where the output should be placed.
        Returns:
            pos_emb: [C, hidden_dim]
        """
        device = device or self.asc_vv_param.device 

        try:
            embeddings = torch.stack(
                [self.pol_map[pol.lower()].to(device) for pol in polarizations],
                dim=0
            )
        except KeyError as e:
            raise ValueError(f"Unknown polarization: {e.args[0]}. "
                             f"Valid options: {list(self.pol_map.keys())}")

        return self.mlp(embeddings)                  # [C, embed_dim]

class DemProjector(nn.Module):

    def __init__(self, embed_dim=768, hidden_dim=768):

        super().__init__()
        self.embed_dim = embed_dim

        self.dsm_param = nn.Parameter(torch.empty(embed_dim))
        self.dtm_param = nn.Parameter(torch.empty(embed_dim))
        self.slope_param = nn.Parameter(torch.empty(embed_dim))

        for name, param in self.named_parameters():
            if "param" in name:
                nn.init.normal_(param, 0.0, 0.02)


        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.dem_map = {
            "dsm": self.dsm_param,
            "dtm": self.dtm_param,
            "slope": self.slope_param
        }

    def forward(self, polarizations: list, device=None) -> torch.Tensor:
        """
        Args:
            polarizations: List of polarization strings ['vv', 'vh', 'hh', 'hv'] of length [C].
        Returns:
            pos_emb: [C, hidden_dim]
        """
        embeddings = []
        device = device or self.dsm_param.device 

        try:
            embeddings = torch.stack(
                [self.dem_map[pol.lower()].to(device) for pol in polarizations],
                dim=0
            )
        except KeyError as e:
            raise ValueError(f"Unknown model: {e.args[0]}. "
                             f"Valid options: {list(self.dem_map.keys())}")

        return self.mlp(embeddings)      


class CrossAttention(nn.Module):
    """
    Cross attention module.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.q_norm = norm_layer(self.head_dim)
        self.k_norm = norm_layer(self.head_dim)

        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, x_resampled):

        q = self.q_proj(x_resampled)
        k = self.k_proj(x)
        v = self.v_proj(x)   

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        q, k = self.q_norm(q), self.k_norm(k)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        out = rearrange(out, 'b h n d -> b n (h d)') 
        out = self.proj(out)

        return out
        
class AttentiveResampler(nn.Module):
    """
    Attention-based resampling module.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 norm_layer: nn.Module = nn.LayerNorm,
                 q_size: int = 16):
        super().__init__()

        self.q_size = q_size

        self.cross_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            norm_layer=norm_layer,
        )

    def forward(self, x, gsd, gsd_target):

        B, C, H, W = x.shape
        ratio = gsd / gsd_target
        new_H = int(H * ratio)
        new_W = int(W * ratio)

        x_resampled = F.interpolate(
            x,
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False,
        )

        q_tiles = new_H // self.q_size
        kv_tiles = H // self.q_size
        x_resampled = rearrange(
            x_resampled,
            'b c (n1 h) (n2 w) -> b (n1 n2) (h w) c',
            n1=q_tiles,
            n2=q_tiles,
        )
        x = rearrange(
            x,
            'b c (m1 h) (m2 w) -> b (m1 m2) (h w) c',
            m1=kv_tiles,
            m2=kv_tiles,
        )

        

        out = self.cross_attn(x, x_resampled)

        return out
        
        


