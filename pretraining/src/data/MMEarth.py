from torch.utils.data import Dataset
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
import torch
from datetime import datetime
import h5py

def collate_fn(batch):
    """
    Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "label", "name"  and the other corresponding to the modalities used
    Returns:
        dict: dictionary with keys "label", "name"  and the other corresponding to the modalities used
    """
    keys = list(batch[0].keys())
    output = {}
    if 'name' in keys:
        output['name'] = [x['name'] for x in batch]
        keys.remove('name')
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    return output

class MMEarth(Dataset):
    def __init__(self, path, modalities, transform, split="train", classes=[], 
                 partition=1.0, norm_path=None):
        self.path = path
        self.modalities = modalities
        self.transform = transform
        self.split = split
        self.partition = partition

        self.h5_path = os.path.join(path, "data_1M_v001_64.h5")
        self.h5_file = None

        self.mod_mapping = {
            "dem": "aster",
            "s2": "sentinel2",
            "s1": "sentinel1",
        }

        with h5py.File(self.h5_path, "r") as f:
            n_samples = len(f["biome"])
            self.ids = np.arange(n_samples)
            self.labels_oh = np.array(f["biome"])
            self.labels = np.argmax(self.labels_oh, axis=1)

        if self.partition < 1:
            self.ids, _, self.labels, _ = train_test_split(
                self.ids,
                self.labels,
                stratify=self.labels,
                test_size=1 - self.partition,
            )

        if self.split in ["train", "val"]:
            train_ids, val_ids, train_labels, val_labels = train_test_split(
                self.ids,
                self.labels,
                stratify=self.labels,
                test_size=0.1,
                random_state=42,
            )
            if self.split == "train":
                self.ids, self.labels = train_ids, train_labels
            else:
                self.ids, self.labels = val_ids, val_labels

        self.collate_fn = collate_fn
        self.norm = None

        if norm_path is not None:
            norm = {}
            for modality in self.modalities:
                if modality == "s2":
                    file_path_1 = os.path.join(norm_path, f"NORM_{modality}_l1c_patch.json")
                    file_path_2= os.path.join(norm_path, f"NORM_{modality}_l2a_patch.json")
                    normvals_1 = json.load(open(file_path_1))
                    normvals_2 = json.load(open(file_path_2))
                    norm[f"{modality}_l1c"] = (
                        torch.tensor(normvals_1["mean"]).float(),
                        torch.tensor(normvals_1["std"]).float(),
                    )
                    norm[f"{modality}_l2a"] = (
                        torch.tensor(normvals_2["mean"]).float(),
                        torch.tensor(normvals_2["std"]).float(),
                    )
                else:
                    file_path = os.path.join(norm_path, f"NORM_{modality}_patch.json")
                    normvals = json.load(open(file_path))
                    norm[modality] = (
                        torch.tensor(normvals["mean"]).float(),
                        torch.tensor(normvals["std"]).float(),
                    )
            self.norm = norm

    def compute_norm_vals(self, folder, sat, max_files=None, sample_ratio=1.0):
        means, stds = [], []
        for i in range(len(self.ids)):
            data = self.__getitem__(i)[sat]
            if len(data.shape) == 4:  # (T, C, H, W)
                data = data.permute(1, 0, 2, 3)
                means.append(data.to(torch.float32).mean(dim=(1, 2, 3)).numpy())
                stds.append(data.to(torch.float32).std(dim=(1, 2, 3)).numpy())
            else:  # (C, H, W)
                means.append(data.to(torch.float32).mean(dim=(1, 2)).numpy())
                stds.append(data.to(torch.float32).std(dim=(1, 2)).numpy())
        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)
        norm_vals = dict(mean=list(mean), std=list(std))
        with open(os.path.join(folder, f"NORM_{sat}_patch.json"), "w") as f:
            json.dump(norm_vals, f, indent=4)

    def _ensure_h5_open(self):
        if self.h5_file is None:
            # open file once per worker
            self.h5_file = h5py.File(self.h5_path, "r")
        return self.h5_file

    def __getitem__(self, i):
        
        f = self._ensure_h5_open()
        idx = self.ids[i]
        output = {"label": torch.tensor(self.labels[i]), "name": torch.tensor(idx)}
        for modality in self.modalities: 
            mod = self.mod_mapping[modality]
            data = torch.tensor(f[mod][idx]).unsqueeze(0)
            data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            output[modality] = data
            output[f"{modality}_dates"] = torch.tensor(0)
                
        if self.norm is not None:
            for modality in self.modalities:
                data = output[modality]
    
                if torch.all(data == 0):
                    continue
    
                if modality == "s2":
                    band11 = data[..., 10, :, :] if data.ndim == 4 else data[10, :, :]
                    is_l2a = torch.all(band11 == 0)
                    norm_key = "s2_l2a" if is_l2a else "s2_l1c"
                    mean, std = self.norm[norm_key]
                else:
                    mean, std = self.norm[modality]
    
                
                mask = torch.any(data != 0, dim=(-1, -2), keepdim=True)  # [..., C, 1, 1]
                if data.ndim == 4: 
                    mean_ = mean[None, :, None, None]
                    std_ = std[None, :, None, None]
                else: 
                    mean_ = mean[:, None, None]
                    std_ = std[:, None, None]
    
                data = torch.where(mask, (data - mean_) / std_, data)
                output[modality] = data
                
        return self.transform(output)

    def __len__(self):
        return len(self.ids)
