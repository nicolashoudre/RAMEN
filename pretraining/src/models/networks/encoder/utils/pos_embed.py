#https://github.com/microsoft/torchgeo/blob/main/torchgeo/models/scale_mae.py
import torch
from torch import Tensor

def get_2d_sincos_pos_embed_with_resolution(
    embed_dim: int, grid_size: int, res: Tensor, cls_token: bool = False
) -> Tensor:
    """Generate spatial resolution specific 2D positional embeddings.

    Args:
        embed_dim: Dimension of the positional embeddings.
        grid_size: Height (ph) and width (pw) of the image patches.
        res: Spatial resolution tensor of shape (N,) of the image.
        cls_token: Increase positional embedding size by 1 for class token.

    Returns:
        pos_embed: Spatial resolution aware positional embeddings (Ph * Pw, D).
    """
    device, dtype = res.device, res.dtype
    grid_h = torch.arange(grid_size, dtype=dtype, device=device)
    grid_w = torch.arange(grid_size, dtype=dtype, device=device)
    grid: Tensor = torch.stack(torch.meshgrid(grid_w, grid_h, indexing='xy'), dim=0)
    grid = torch.einsum('chw,n->cnhw', grid, res)
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    if cls_token:
        pos_embed = torch.cat(
            [torch.zeros([n, 1, embed_dim], dtype=dtype, device=device), pos_embed],
            dim=1,
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid_torch(embed_dim: int, grid: Tensor) -> Tensor:
    """Generate 2D sin-cos positional embedding from grid.

    Args:
        embed_dim: Dimension of the positional embeddings.
        grid: Tensor representing the image patch grid (C, N, Ph, Pw)

    Returns:
        emb: 2D sin-cos positional embeddings (Ph * Pw, D).
    """
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[1])
    emb = torch.cat([emb_h, emb_w], dim=1)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim: int, pos: Tensor) -> Tensor:
    """Generate 1D sin-cos positional embedding from grid dimension.

    Args:
        embed_dim: Dimension of the positional embeddings.
        pos: Tensor of positions to be encoded (M,).

    Returns:
        emb: 1D sin-cos positional embeddings (M, D).
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=pos.dtype, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb