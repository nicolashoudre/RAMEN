import torch
from torch import nn
from torch.nn import functional as F
from models.networks.encoder.utils.ramen_utils import ScaleResampler, SpectralProjector, RadarProjector, DemProjector
from models.networks.encoder.utils.pos_embed import get_2d_sincos_pos_embed_with_resolution, get_1d_sincos_pos_embed_from_grid_torch
from timm.models.vision_transformer import Block
from timm.layers.attention_pool import AttentionPoolLatent
from models.networks.encoder.utils.utils import trunc_normal_
import random
from einops import rearrange
from collections import defaultdict
    
class RamenDecoderViT(nn.Module):
    """
    Initialize RAMEN decoder with multiple resolutions per modality and multiple datasets handling
    Args:
        datasets (list): List of datasets to be processed
        modalities (dict): dict with available modalities for each dataset
        modalities_chans (dict): nested dict mapping datasets to their modalities and channel sizes
        modalities_res (dict): nested dict mapping datasets to their modalities and resolutions
        modalities_input_size (dict): nested dict mapping datasets to their modalities and input sizes
        all_res (dict): dict mapping datasets to their all resolutions to be used
    """
    def __init__(
        self,
        datasets: list = [],
        modalities: dict = {},
        modalities_res: dict = {},
        modalities_input_size: dict = {},
        all_res: dict = {},
        wavelengths: dict = {},
        embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qvk_bias: bool = True,
        qk_scale: float = None,
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
        self.decoder_embed_dim = decoder_embed_dim

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.spectral_projector = SpectralProjector(decoder_embed_dim, decoder_embed_dim*2)
        if any(value in ["s1_des", "s1", "alos"] for sublist in self.modalities.values() for value in sublist):
            self.radar_projector = RadarProjector(decoder_embed_dim, decoder_embed_dim*2)
        if any(value in ["dem"] for sublist in self.modalities.values() for value in sublist):
            self.dem_projector = DemProjector(decoder_embed_dim, decoder_embed_dim*2)

        self.resampler = ScaleResampler(decoder_embed_dim)

        self.temporal_block = Block(
            dim=decoder_embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qvk_bias,
            qk_norm=qk_scale,
            proj_drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
        )
        self.temporal_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)

        self.blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim,
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

        self.norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)

        self.dec_pos_embed_dict = defaultdict(dict)
        for dataset in datasets:
            for res in self.all_res[dataset]:
                effective_size = int(modalities_input_size[dataset][modalities[dataset][0]] * (modalities_res[dataset][modalities[dataset][0]] / res))
                self.dec_pos_embed_dict[dataset][res] = get_2d_sincos_pos_embed_with_resolution(
                    decoder_embed_dim,
                    effective_size,
                    torch.tensor([res]),
                    cls_token=True,
                )

    def forward(self, out, ids_restore):

        """
        Forward pass of the network
        """
        tokens = self.decoder_embed(out['tokens'])

        mask_tokens = self.mask_token.repeat(tokens.shape[0], ids_restore.shape[1] + 1 - tokens.shape[1], 1)
        tokens_ = torch.cat([tokens[:, 1:, :], mask_tokens], dim=1)  # no cls token
        tokens_ = torch.gather(tokens_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, tokens.shape[2]))  # unshuffle
        tokens = torch.cat([tokens[:, :1, :], tokens_], dim=1)  # append cls token

        pos_embed = self.dec_pos_embed_dict[out['dataset']][out['effective_res']].to(device=tokens.device, dtype=tokens.dtype)
        pos_embed = torch.cat((pos_embed[:, :1, :], pos_embed[:, 1:, :].repeat(1, len(out['kept_modalities']), 1)), dim=1)
        tokens = tokens + pos_embed

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        
        tokens = tokens[:, 1:, :]  # remove cls token
        
        B, N, D = tokens.shape
        N_mod = N // len(out['kept_modalities'])
        modality_tokens = {
            m: tokens[:, i * N_mod:(i + 1) * N_mod, :]
            for i, m in enumerate(out['kept_modalities'])
        }
        H = W = int(N_mod ** 0.5)
        device, dtype = tokens.device, tokens.dtype
        
        for modality in out['kept_modalities']:
            T = out[f'{modality}_TS']
            mod_tokens = modality_tokens[modality]  # [B, H*W, D]
            if T > 1:
                dates = out[f'{modality}_dates'].float()
                temporal_embed = torch.stack([
                    get_1d_sincos_pos_embed_from_grid_torch(D, dates[b])
                    for b in range(B)
                ], dim=0).to(device=tokens.device, dtype=tokens.dtype) # [B, T, D]
                temporal_embed = temporal_embed.unsqueeze(1).expand(B, H*W, T, D)
                mod_tokens = mod_tokens.unsqueeze(2).expand(-1, -1, T, -1)
                out_mod = (mod_tokens + temporal_embed).reshape(B * H * W, T, D)
                out_mod = self.temporal_block(out_mod)
                out_mod = self.temporal_norm(out_mod)
                out_mod = out_mod.reshape(B, H * W, T, D).permute(0, 2, 3, 1).reshape(B * T, D, H, W)  # [B*T, D, H, W]
            else:
                out_mod = mod_tokens.transpose(1, 2).reshape(B, D, H, W)  # [B, D, H, W]
        
            scale = out['effective_res'] / self.modalities_res[out['dataset']][modality]
            expected_size = self.modalities_input_size[out['dataset']][modality]
            out_mod = self.resampler(out_mod, scale)
        
            if out_mod.shape[2] != expected_size or out_mod.shape[3] != expected_size:
                out_mod = F.interpolate(out_mod, size=(expected_size, expected_size), mode='bilinear')
        
            out_mod = out_mod.view(B, T, -1, expected_size, expected_size).permute(0, 1, 3, 4, 2).contiguous()  # [B, T, H, W, D]
        
            if modality in ["s1", "alos", "s1_des", "s1_asc"]:
                spectral_encoding = self.radar_projector(self.wavelengths[out['dataset']][modality], device)
            elif modality in ["dem"]:
                spectral_encoding = self.dem_projector(self.wavelengths[out['dataset']][modality], device)
            else:
                spectral_encoding = self.spectral_projector(
                    torch.tensor(self.wavelengths[out['dataset']][modality], device=device, dtype=dtype)
                )
        
            out[f'{modality}_reconstructed'] = out_mod @ spectral_encoding.transpose(0, 1)
            
        return out
