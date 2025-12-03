import torch
from torch import nn
from torch.nn import functional as F
from hydra.utils import instantiate

class RAMENMAE(nn.Module):
    """
    Initialize encoder and decoder
    Args:
        encoder (nn.Module): that encodes data
        decoder (nn.Module): that decodes data
    """
    def __init__(
        self,
        encoder,
        decoder,
        mask_ratio = 0.75
    ):
        
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        x = x['tokens']
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, effective_res, modalities_keep=None):
        """
        Forward pass of the network
        """
        out = self.encoder.forward_proj(x, effective_res, modalities_keep)
        out['tokens'], mask, ids_restore = self.random_masking(out, self.mask_ratio)
        out['tokens'] = self.encoder.forward_transformer(out)

        return out, mask, ids_restore
    
    def forward_decoder(self, out, ids_restore):
        """
        Forward pass of the network
        """
        # embed tokens
        out = self.decoder(out, ids_restore)
        return out

    def forward(self, x, effective_res, modalities_keep=None):
        """
        Forward pass of the network
        """
        # -- forward encoder
        out, mask, ids_restore = self.forward_encoder(x, effective_res, modalities_keep)
        out = self.forward_decoder(out, ids_restore)
        return out, mask

