import torch
import torch.nn as nn
from models.networks.encoder.utils.vit_utils import AttentionPoolLatent
from einops import rearrange
import functools

class Fine(nn.Module):
    """
    Initialize Fine Tuning of OmniSat after pretraining
    Args:
        encoder (torch.nn.Module): initialized model
        path (str): path of checkpoint of model to load
        output_size (int): size of output returned by encoder
        inter_dim (list): list of hidden dims of mlp after encoder
        p_drop (float): dropout parameter of mlp after encoder
        name (str): name of the weights from checkpoint to use
        freeze (bool); if True, freeze encoder to perform linear probing
        n_class (int): output_size of mlp
        pooling_method (str): type of pooling of tokens after transformer
        modalities (list): list of modalities to use
        last_block (bool): if True freeze all encoder except last block of transformer
        proj_only (bool): if True, load only weights from projectors
    """
    def __init__(self, 
                 encoder: torch.nn.Module,
                 path: str = '',
                 output_size: int = 256,
                 inter_dim: list = [],
                 p_drop: float = 0.3,
                 name: str = 'encoder',
                 freeze: bool = True,
                 n_class: int = 15,
                 pooling_method: str = 'token',
                 modalities: list = [],
                ):
        super().__init__()

        self.size = output_size
        self.freeze = freeze
        self.global_pool = pooling_method
        self.modalities = modalities

        #for i in range(len(modalities)):
        #    if modalities[i].split('-')[-1] == 'mono':
        #        modalities[i] = '-'.join(modalities[i].split('-')[:-1])

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
        
            save_path = "/lustre/fsn1/projects/rech/qsm/uix17xe/filtered_encoder.ckpt"
            torch.save({"state_dict": encoder.state_dict()}, save_path)
            print(f"Filtered encoder weights saved to {save_path}")
        
            del u
            del d

                 
        self.encoder = encoder
        
        
        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        self.n_class = n_class
        # set n_class to 0 if we want headless model
        if n_class:
            layers = [nn.LayerNorm(self.size)]
            if len(inter_dim) > 0:
                layers.append(nn.Linear(self.size, inter_dim[0]))
                #layers.append(nn.BatchNorm1d(inter_dim[0]))
                layers.append(nn.Dropout(p = p_drop))
                layers.append(nn.ReLU())
                for i in range(len(inter_dim) - 1):
                    layers.append(nn.Linear(inter_dim[i], inter_dim[i + 1]))
                    #layers.append(nn.BatchNorm1d(inter_dim[i + 1]))
                    layers.append(nn.Dropout(p = p_drop))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(inter_dim[-1], n_class))
            else:
                layers.append(nn.Linear(self.size, n_class))
            self.head = nn.Sequential(*layers)

        if self.global_pool == 'attn':
            self.AttentionPooling = AttentionPoolLatent(in_features=self.size, num_heads=4)
        if self.global_pool == 'mod_avg':
            self.modality_head = nn.Linear(len(self.modalities)*self.size, self.size)
        
    def forward(self, x):
        """
        Forward pass of the network. Perform pooling of tokens after transformer 
        according to global_pool argument.
        """
        effective_res = {mod: self.encoder.all_res[0] for mod in self.modalities}
        out = self.encoder(x, effective_res)
        x = out['tokens']
        if self.global_pool:
            if self.global_pool == 'avg':
                if self.encoder.cls_token is not None:
                    x = x[:, 1:].mean(dim=1)
                else:
                    x = x.mean(dim=1)
            elif self.global_pool == 'avg_mod1':
                if self.encoder.cls_token is not None:
                    x = x[:, 1:]
                n_per_mod = x.shape[1]//2
                x = x[:, :n_per_mod].mean(dim=1)
            elif self.global_pool == 'max':
                if self.encoder.cls_token is not None:
                    x ,_ = torch.max(x[:, 1:],1)
                else:
                    x ,_ = torch.max(x,1)
            elif self.global_pool == 'attn':
                if self.encoder.cls_token is not None:
                    x = x[:, 1:]
                x = self.AttentionPooling(x)
            elif self.global_pool == 'mod_avg':
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
            if self.n_class:
                x = self.head(x)   
            return x
        return x
    
class FineSemSeg(nn.Module):
    """
    Initialize Fine Tuning of OmniSat after pretraining
    Args:
        encoder (torch.nn.Module): initialized model
        path (str): path of checkpoint of model to load
        output_size (int): size of output returned by encoder
        inter_dim (list): list of hidden dims of mlp after encoder
        p_drop (float): dropout parameter of mlp after encoder
        name (str): name of the weights from checkpoint to use
        freeze (bool); if True, freeze encoder to perform linear probing
        n_class (int): output_size of mlp
        pooling_method (str): type of pooling of tokens after transformer
        modalities (list): list of modalities to use
        last_block (bool): if True freeze all encoder except last block of transformer
        proj_only (bool): if True, load only weights from projectors
    """
    def __init__(self, 
                 encoder: torch.nn.Module,
                 path: str = '',
                 pred_size: int = 512,
                 output_size: int = 256,
                 inter_dim: list = [],
                 name: str = 'encoder',
                 freeze: bool = True,
                 n_class: int = 15,
                 use_attn_pooling: bool = False,
                 modalities: list = [],
                 decoder_channels: dict = None,
                ):
        super().__init__()

        self.size = output_size
        self.pred_size = pred_size
        self.freeze = freeze
        self.modalities = modalities

        for i in range(len(modalities)):
            if modalities[i].split('-')[-1] == 'mono':
                modalities[i] = '-'.join(modalities[i].split('-')[:-1])

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
        
        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        self.n_class = n_class
        self.use_attn_pooling = use_attn_pooling

        self.modality_head = nn.ModuleDict()
        for r in self.encoder.all_res:
            if use_attn_pooling is True:
                self.modality_head[f"res_{str(r).replace('.', '_')}"] = nn.Sequential(
                    AttentionPoolLatent(in_features=self.size, num_heads=4),
                )
            else:
                self.modality_head[f"res_{str(r).replace('.', '_')}"] = nn.Sequential(
                    nn.Linear(len(self.modalities)*self.size, self.size),
                )

        sorted_res = sorted(self.encoder.all_res, reverse=True)
        if decoder_channels is None:
            decoder_channels = {}
            for i, r in enumerate(sorted_res):
                decoder_channels[r] = max(self.size//(2**i), n_class)
        
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(sorted_res) - 1):
            in_chans, out_chans = decoder_channels[sorted_res[i]] + decoder_channels[sorted_res[i+1]], decoder_channels[sorted_res[i+1]]
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            )

        self.res_proj = nn.ModuleDict()
        for r in sorted_res:
            self.res_proj[f"res_{str(r).replace('.', '_')}"] = nn.Conv2d(self.size, decoder_channels[r], kernel_size=1)

        self.final_conv = nn.Conv2d(decoder_channels[min(sorted_res)], n_class, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass of the network. Perform pooling of tokens after transformer 
        according to global_pool argument.
        """
        
        M = len(self.modalities)
        D = self.size
        features_dict = {}
        sorted_res = sorted(self.encoder.all_res, reverse=True) 
        x_prev = None
        
        for i, r in enumerate(sorted_res):
            effective_res = {mod: r for mod in self.modalities}
            out = self.encoder(x, effective_res)
            x_r = out['tokens']
            B = x_r.shape[0]
            
            N = x_r.shape[1]
            P = N // M
            H = W = int(P ** 0.5)
            
            x_r = rearrange(x_r, 'b (m h w) d -> b h w m d', m=M, h=H, w=W)
            if self.use_attn_pooling is True:
                x_r = rearrange(x_r, 'b h w m d -> (b h w) m d')
                x_r = self.modality_head[f"res_{str(r).replace('.', '_')}"](x_r)  
                x_r = rearrange(x_r, '(b h w) d -> b h w d', b=B, h=H, w=W)
            else:  
                x_r = rearrange(x_r, 'b h w m d -> b h w (m d)')
                x_r = self.modality_head[f"res_{str(r).replace('.', '_')}"](x_r)  # [B, H, W, D]
            x_r = rearrange(x_r, 'b h w d -> b d h w')
            x_r = self.res_proj[f"res_{str(r).replace('.', '_')}"](x_r)
            
            if x_prev is None:
                x_prev = x_r
            else:
                x_up = nn.functional.interpolate(x_prev, size=x_r.shape[-2:], mode='bilinear', align_corners=False)
                x_cat = torch.cat([x_up, x_r], dim=1)
                x_prev = self.decoder_blocks[i-1](x_cat)

        if x_prev.shape[-2:] != (self.pred_size, self.pred_size):
            #while H < self.pred_size or W < self.pred_size: #progressive upsampling for memory issues on large images
            #    H, W = x_prev.shape[-2:]
            #    new_H, new_W = min(H*2, self.pred_size), min(W*2, self.pred_size)
            #    x_prev = nn.functional.interpolate(x_prev, size=(new_H, new_W), mode='bilinear', align_corners=False)
            #    H, W = new_H, new_W
            x_prev = nn.functional.interpolate(x_prev, size=(self.pred_size, self.pred_size), mode='bilinear', align_corners=False)

        pred = self.final_conv(x_prev)
        return pred #[B, n_class, H, W]

class FineSemSegMultiData(nn.Module):
    """
    Initialize Fine Tuning of OmniSat after pretraining
    Args:
        encoder (torch.nn.Module): initialized model
        path (str): path of checkpoint of model to load
        output_size (int): size of output returned by encoder
        inter_dim (list): list of hidden dims of mlp after encoder
        p_drop (float): dropout parameter of mlp after encoder
        name (str): name of the weights from checkpoint to use
        freeze (bool); if True, freeze encoder to perform linear probing
        n_class (int): output_size of mlp
        pooling_method (str): type of pooling of tokens after transformer
        modalities (list): list of modalities to use
        last_block (bool): if True freeze all encoder except last block of transformer
        proj_only (bool): if True, load only weights from projectors
    """
    def __init__(self, 
                 encoder: torch.nn.Module,
                 path: str = '',
                 pred_size: int = 512,
                 output_size: int = 256,
                 inter_dim: list = [],
                 name: str = 'encoder',
                 freeze: bool = True,
                 n_class: int = 15,
                 use_attn_pooling: bool = False,
                 modalities: list = [],
                 decoder_channels: dict = None,
                ):
        super().__init__()

        self.size = output_size
        self.pred_size = pred_size
        self.freeze = freeze
        self.modalities = modalities

        for i in range(len(modalities)):
            if modalities[i].split('-')[-1] == 'mono':
                modalities[i] = '-'.join(modalities[i].split('-')[:-1])

        target = (name == "target_encoder")
        
        if path is not None:
            u = torch.load(path)
            d = {}
            for key in u["state_dict"].keys():
                if name in key:
                    if target:
                        if 'pixel_encoders' in key:
                            if any([modality in key for modality in modalities]):
                                d['.'.join(key.split('.')[1:])] = u["state_dict"][key]
                        else:
                            if not('predictor.' in key):
                                d['.'.join(key.split('.')[1:])] = u["state_dict"][key]
    
            load_result = encoder.load_state_dict(d, strict=False)
    
            print("Missing keys (not loaded):")
            print(load_result.missing_keys)
            
            print("Unexpected keys (in checkpoint but not in model):")
            print(load_result.unexpected_keys)
    
            del u
            del d
                 
        self.encoder = encoder
        
        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        self.n_class = n_class
        self.use_attn_pooling = use_attn_pooling

        self.modality_head = nn.ModuleDict()
        for r in self.encoder.all_res:
            if use_attn_pooling is True:
                self.modality_head[f"res_{str(r).replace('.', '_')}"] = nn.Sequential(
                    AttentionPoolLatent(in_features=self.size, num_heads=4),
                )
            else:
                self.modality_head[f"res_{str(r).replace('.', '_')}"] = nn.Sequential(
                    nn.Linear(len(self.modalities)*self.size, self.size),
                )

        sorted_res = sorted(self.encoder.all_res, reverse=True)
        if decoder_channels is None:
            decoder_channels = {}
            for i, r in enumerate(sorted_res):
                decoder_channels[r] = max(self.size//(2**i), n_class)
        
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(sorted_res) - 1):
            in_chans, out_chans = decoder_channels[sorted_res[i]] + decoder_channels[sorted_res[i+1]], decoder_channels[sorted_res[i+1]]
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            )

        self.res_proj = nn.ModuleDict()
        for r in sorted_res:
            self.res_proj[f"res_{str(r).replace('.', '_')}"] = nn.Conv2d(self.size, decoder_channels[r], kernel_size=1)

        self.final_conv = nn.Conv2d(decoder_channels[min(sorted_res)], n_class, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass of the network. Perform pooling of tokens after transformer 
        according to global_pool argument.
        """
        
        M = len(self.modalities[x['dataset']])
        D = self.size
        features_dict = {}
        sorted_res = sorted(self.encoder.all_res[x['dataset']], reverse=True) 
        x_prev = None
        
        for i, r in enumerate(sorted_res):
            effective_res = {mod: r for mod in self.modalities[x['dataset']]}
            out = self.encoder(x, effective_res)
            x_r = out['tokens']
            B = x_r.shape[0]
            
            N = x_r.shape[1]
            P = N // M
            H = W = int(P ** 0.5)
            
            x_r = rearrange(x_r, 'b (m h w) d -> b h w m d', m=M, h=H, w=W)
            if self.use_attn_pooling is True:
                x_r = rearrange(x_r, 'b h w m d -> (b h w) m d')
                x_r = self.modality_head[f"res_{str(r).replace('.', '_')}"](x_r)  
                x_r = rearrange(x_r, '(b h w) d -> b h w d', b=B, h=H, w=W)
            else:  
                x_r = rearrange(x_r, 'b h w m d -> b h w (m d)')
                x_r = self.modality_head[f"res_{str(r).replace('.', '_')}"](x_r)  # [B, H, W, D]
            x_r = rearrange(x_r, 'b h w d -> b d h w')
            x_r = self.res_proj[f"res_{str(r).replace('.', '_')}"](x_r)
            
            if x_prev is None:
                x_prev = x_r
            else:
                x_up = nn.functional.interpolate(x_prev, size=x_r.shape[-2:], mode='bilinear', align_corners=False)
                x_cat = torch.cat([x_up, x_r], dim=1)
                x_prev = self.decoder_blocks[i-1](x_cat)

        if x_prev.shape[-2:] != (self.pred_size, self.pred_size):
            #while H < self.pred_size or W < self.pred_size: #progressive upsampling for memory issues on large images
            #    H, W = x_prev.shape[-2:]
            #    new_H, new_W = min(H*2, self.pred_size), min(W*2, self.pred_size)
            #    x_prev = nn.functional.interpolate(x_prev, size=(new_H, new_W), mode='bilinear', align_corners=False)
            #    H, W = new_H, new_W
            x_prev = nn.functional.interpolate(x_prev, size=(self.pred_size, self.pred_size), mode='bilinear', align_corners=False)

        pred = self.final_conv(x_prev)
        return pred #[B, n_class, H, W]

class ScaleCheck(nn.Module):
    """
    Initialize Fine Tuning of OmniSat after pretraining
    Args:
        encoder (torch.nn.Module): initialized model
        path (str): path of checkpoint of model to load
        output_size (int): size of output returned by encoder
        inter_dim (list): list of hidden dims of mlp after encoder
        p_drop (float): dropout parameter of mlp after encoder
        name (str): name of the weights from checkpoint to use
        freeze (bool); if True, freeze encoder to perform linear probing
        n_class (int): output_size of mlp
        pooling_method (str): type of pooling of tokens after transformer
        modalities (list): list of modalities to use
        last_block (bool): if True freeze all encoder except last block of transformer
        proj_only (bool): if True, load only weights from projectors
    """
    def __init__(self, 
                 encoder: torch.nn.Module,
                 path: str = '',
                 name: str = 'encoder',
                 freeze: bool = True,
                 n_class: int = 15,
                 use_attn_pooling: bool = False,
                 modalities: list = [],
                ):
        super().__init__()

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
        
        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        self.n_class = n_class
        self.use_attn_pooling = use_attn_pooling

        self.resampler = self.encoder.resampler
        
    def forward(self, x):
        """
        Forward pass of the network. Perform pooling of tokens after transformer 
        according to global_pool argument.
        """
        
        #scales = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
        scales = torch.logspace(-2, 2, steps=100)
        print(f"Temperature: {self.resampler.temperature}")
        
        with torch.no_grad():
            for scale in scales:
                scale_tensor = torch.tensor([scale], device=x["aerial"].device, dtype=x["aerial"].dtype)
                scale_emb = self.resampler.scale_encoding(scale_tensor)
                #print(f"Scale={scale:.2f} -> Embedding min={scale_emb.min().item():.4f}, "
      #f"max={scale_emb.max().item():.4f}, mean={scale_emb.mean().item():.4f}, "
      #f"std={scale_emb.std().item():.4f}")
                weights = self.resampler.mlp(scale_emb).softmax(dim=-1)
                print(f"Scale={scale:.4f} -> Softmax Weights: {weights.squeeze().tolist()}")
        return scales

if __name__ == "__main__":
    _ = Fine()