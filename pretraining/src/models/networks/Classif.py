import torch
import torch.nn as nn
from models.networks.encoder.utils.vit_utils import AttentionPoolLatent
from einops import rearrange
import functools

class Classif(nn.Module):
    """
    Initialize classif
    Args:
        encoder (torch.nn.Module): initialized model
        path (str): path of checkpoint of model to load
        name (str): name of the weights from checkpoint to use
        freeze (bool); if True, freeze encoder to perform linear probing
        num_classes (int): output_size of mlp
        pooling_method (str): type of pooling of tokens after transformer
        modalities (list): list of modalities to use
    """
    def __init__(self, 
                 encoder: torch.nn.Module,
                 path: str = '',
                 inter_dim: list = [],
                 name: str = 'encoder',
                 freeze: bool = True,
                 num_classes: int = 15,
                 pooling_method: str = 'cls',
                 modalities: list = [],
                ):
        super().__init__()

        self.freeze = freeze
        self.global_pool = pooling_method
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
        
            print(f"Loaded {len(d)} filtered keys into encoder.")
        
            del u
            del d

                 
        self.encoder = encoder
        
        
        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        self.num_classes = num_classes
        if num_classes:
            self.head = nn.Linear(self.encoder.embed_dim, num_classes)

        if self.global_pool == 'attn':
            self.AttentionPooling = AttentionPoolLatent(in_features=self.encoder.embed_dim, num_heads=4)
        if self.global_pool == 'linear':
            self.modality_head = nn.Linear(len(self.modalities)*self.encoder.embed_dim, self.encoder.embed_dim)
        
    def forward(self, x):
        """
        Forward pass of the network. Perform pooling of tokens after transformer 
        according to global_pool argument.
        """
        effective_res = self.encoder.all_res[0] 
        out = self.encoder(x, effective_res)
        x = out['tokens']
        if self.global_pool:
            if self.global_pool == 'avg':
                if self.encoder.cls_token is not None:
                    x = x[:, 1:].mean(dim=1)
                else:
                    x = x.mean(dim=1)
            elif self.global_pool == 'attn':
                if self.encoder.cls_token is not None:
                    x = x[:, 1:]
                x = self.AttentionPooling(x)
            elif self.global_pool == 'linear':
                M = len(self.modalities)
                N = x.shape[1]
                P = N // M
                H = W = int(P ** 0.5)
                if self.encoder.cls_token is not None:
                    x = x[:, 1:]
                x = rearrange(x, 'b (m h w) d -> b (h w) (m d)', m=M, h=H, w=W)
                x = self.modality_head(x)
                x = x.mean(dim=1)
            else:
                x = x[:, 0]
            if self.num_classes:
                x = self.head(x)   
        return x


if __name__ == "__main__":
    _ = Classif()
