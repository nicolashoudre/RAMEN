import torch

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.networks.encoder.utils.pos_embed import get_1d_sincos_pos_embed_from_grid_torch

class ScaleEncoding(nn.Module):
    def __init__(self, embed_dim: int):
        """
        Args:
            embed_dim: Dimension of the positional encoding.
        """
        super().__init__()
        assert embed_dim % 2 == 0, "Embedding dimension must be even"
        self.embed_dim = embed_dim

    def forward(self, scales: Tensor) -> Tensor:
        """
        Args:
            wavelengths: Tensor of spectral positions [C].
                        e.g., scales of each input channel.
        Returns:
            pos_emb: [C, embed_dim]
        """

        scales = torch.log(scales)
        pos_emb = get_1d_sincos_pos_embed_from_grid_torch(self.embed_dim, scales)
        
        return pos_emb

class LowRankConv(nn.Module):
    def __init__(self, embed_dim: int, rank: int):
        """
        Low-rank convolution
        """
        super().__init__()
        self.U = nn.Conv2d(embed_dim, rank, kernel_size=1, bias=False)  
        self.V = nn.Conv2d(rank, embed_dim, kernel_size=1, bias=True) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.V(self.U(x))

class ScaleResampler(nn.Module):
    
    def __init__(self, embed_dim: int, num_experts: int = 4):

        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            #LowRankConv(embed_dim, embed_dim // 2) for _ in range(num_experts)
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1) for _ in range(num_experts)
        ])

        self.scale_encoding = ScaleEncoding(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, num_experts),
        )
        self.temperature = nn.Parameter(torch.tensor(0.7))

    def forward(self, x, scale):
        
        x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)

        scale_tensor = torch.tensor([scale], device=x.device, dtype=x.dtype)
        scale_emb = self.scale_encoding(scale_tensor)

        weights = (self.mlp(scale_emb) / self.temperature.clamp(min=1e-3)).softmax(dim=-1)    

        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out += weights[0, i] * expert(x)
        return x + out
        
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


