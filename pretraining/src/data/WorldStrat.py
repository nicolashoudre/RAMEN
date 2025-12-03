import os
import json
import torch
import rasterio
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from datetime import datetime
import geopandas as gpd
from skmultilearn.model_selection import iterative_train_test_split

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

    ts_modalities = ["s2", "s1_asc", "s1_des", "s1"]
    for key in ts_modalities:
        if key in keys:
            idx = [x[key] for x in batch]
            max_len = max(tensor.size(0) for tensor in idx)
            stacked = torch.stack([
                torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, 0, 0, max_len - tensor.size(0)))
                for tensor in idx
            ], dim=0)
            output[key] = stacked
            keys.remove(key)
            dates_key = f"{key}_dates"
            if dates_key in batch[0]:
                idx_dates = [x[dates_key] for x in batch]
                max_len_dates = max(t.size(0) for t in idx_dates)
                stacked_dates = torch.stack([
                    torch.nn.functional.pad(t, (0, max_len_dates - t.size(0)))
                    for t in idx_dates
                ], dim=0)
                output[dates_key] = stacked_dates
                if dates_key in keys:
                    keys.remove(dates_key)

    if "name" in keys:
        output["name"] = [x["name"] for x in batch]
        keys.remove("name")

    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])

    return output

class WorldStrat(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        partition: float = 1.0,
        num_classes: int = 19,
        norm_path=None,
        cloud_threshold: float = 100.0,
        temporal_dropout: float = 0.0,
        nb_split: int = 4,
    ):
        self.path = Path(path)
        self.transform = transform
        self.partition = partition
        self.num_classes = num_classes
        self.cloud_threshold = cloud_threshold
        self.temporal_dropout = temporal_dropout
        self.nb_split = nb_split
        self.split = split
        self.collate_fn = collate_fn

        self.modalities = modalities

        metadata = pd.read_csv(self.path / "metadata.csv")
        split_df = pd.read_csv(self.path / "stratified_train_val_test_split.csv")
        if metadata.columns[0] != "ROI":
            metadata = metadata.rename(columns={metadata.columns[0]: "ROI"})
        self.data = metadata.merge(
            split_df, left_on="ROI", right_on="tile", how="inner"
        )
        self.data = self.data[self.data["split"] == split].reset_index(drop=True)

        if partition < 1.0:
            self.data = self.data.sample(frac=partition, random_state=42).reset_index(drop=True)
        self.labels = None
                
        self.norm = None
        if norm_path is not None:
            norm = {}
            for mod in self.modalities:
                file_path = os.path.join(norm_path, f"NORM_{mod}_patch.json")
                if not os.path.exists(file_path):
                    self.compute_norm_vals(norm_path, mod)
                normvals = json.load(open(file_path))
                norm[mod] = (
                    torch.tensor(normvals["mean"]).float(),
                    torch.tensor(normvals["std"]).float(),
                )
            self.norm = norm

    def compute_norm_vals(self, folder, modality):
        means, stds = [], []
        for i in range(self.data["ROI"].nunique() * self.nb_split * self.nb_split):
            data = self.__getitem__(i)[modality]
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
        with open(os.path.join(folder, f"NORM_{modality}_patch.json"), "w") as f:
            json.dump(norm_vals, f, indent=4)

    def _load_s2_timeseries(self, roi):
        """Load all available S2 revisits for one ROI."""
        roi_rows = self.data[self.data["ROI"] == roi]
        s2_list, date_list = [], []

        for _, row in roi_rows.iterrows():
            if row["cloud_cover"] > self.cloud_threshold:
                continue

            tiff_path = self.path / "lr_dataset" / roi / "L2A" / f"{roi}-{row['n']}-L2A_data.tiff"
            if not tiff_path.exists():
                continue

            with rasterio.open(tiff_path) as f:
                data = torch.FloatTensor(f.read())  # (C, H, W)

            data = torch.nn.functional.interpolate(
                data.unsqueeze(0),
                size=(158, 158),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
            s2_list.append(data)

            dt = datetime.strptime(str(row["lowres_date"]), "%Y-%m-%d")
            date_list.append(dt.timetuple().tm_yday)

        if not s2_list:
            raise RuntimeError(f"No valid S2 data for ROI {roi}")

        s2_tensor = torch.stack(s2_list, dim=0)
        date_tensor = torch.tensor(date_list, dtype=torch.int32)

        if self.split == "train" and self.temporal_dropout > 0 and len(date_tensor) > 1:
            N = s2_tensor.shape[0]
            keep_n = max(1, int(N * (1 - self.temporal_dropout)))
            keep_idx = torch.randperm(N)[:keep_n]
            s2_tensor = s2_tensor[keep_idx]
            date_tensor = date_tensor[keep_idx]

        return s2_tensor, date_tensor

    def _load_spot(self, roi, roi_rows):
        """Load high-res SPOT image and its acquisition date."""
        path = self.path / "hr_dataset" / roi / f"{roi}_ps.tiff"
        with rasterio.open(path) as f:
            spot = torch.FloatTensor(f.read())

        spot_date_str = str(roi_rows.iloc[0]["highres_date"])
        dt = datetime.strptime(spot_date_str, "%Y-%m-%d")
        spot_date = torch.tensor([dt.timetuple().tm_yday], dtype=torch.int32)

        return spot.unsqueeze(0), spot_date 
        
    def _split_image(self, image_tensor, nb_split, id):
        if nb_split == 1:
            return image_tensor
        i1 = id // nb_split
        i2 = id % nb_split
        height, width = image_tensor.shape[-2:]
        half_height = height // nb_split
        half_width = width // nb_split
        if image_tensor.dim() == 4:
            return image_tensor[:, :, i1*half_height:(i1+1)*half_height, i2*half_width:(i2+1)*half_width].float()
        if image_tensor.dim() == 3:
            return image_tensor[:, i1*half_height:(i1+1)*half_height, i2*half_width:(i2+1)*half_width].float()
        if image_tensor.dim() == 2:
            return image_tensor[i1*half_height:(i1+1)*half_height, i2*half_width:(i2+1)*half_width].float()

    def __getitem__(self, i):
        roi = self.data["ROI"].unique()[i // (self.nb_split * self.nb_split)]
        roi_rows = self.data[self.data["ROI"] == roi]
        part = i % (self.nb_split * self.nb_split)
        output = {"name": roi}

        output["label"] = torch.tensor([roi_rows.iloc[0]["LCCS"]], dtype=torch.int32)

        # Load modalities
        if "s2" in self.modalities:
            s2, s2_dates = self._load_s2_timeseries(roi)
            s2 = self._split_image(s2, self.nb_split, part)
            output["s2"] = s2
            output["s2_dates"] = s2_dates

        if "spot" in self.modalities:
            spot, spot_date = self._load_spot(roi, roi_rows)
            spot = self._split_image(spot, self.nb_split, part)
            output["spot"] = spot
            output["spot_dates"] = spot_date

        # Normalization
        if self.norm is not None:
            for mod in self.modalities:
                data = output[mod]
                mean, std = self.norm[mod]
                if len(data.shape) == 4:  # (T, C, H, W)
                    output[mod] = (data - mean[None, :, None, None]) / std[None, :, None, None]
                else:  # (C, H, W)
                    output[mod] = (data - mean[:, None, None]) / std[:, None, None]

        if self.transform:
            output = self.transform(output)

        return output

    def __len__(self):
        return self.data["ROI"].nunique() * self.nb_split * self.nb_split

class WorldStrat_PT(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        partition: float = 1.0,
        num_classes: int = 19,
        temporal_dropout: float = 0.0,
        norm_path=None,
        classif: bool = False,
    ):
        """
        Dataset reading pre-saved .pt files per patch.
        """
        self.path = Path(path) / split
        self.split = split
        self.modalities = modalities
        self.transform = transform
        self.partition = partition
        self.num_classes = num_classes
        self.temporal_dropout = temporal_dropout
        self.classif = classif
        self.collate_fn = collate_fn

        all_files = list(self.path.glob("*.pt"))
        if partition < 1.0:
            n = int(len(all_files) * partition)
            self.files = all_files[:n]
        else:
            self.files = all_files

        self.norm = None
        if norm_path is not None:
            norm = {}
            for mod in self.modalities:
                file_path = Path(norm_path) / f"NORM_{mod}_patch.json"
                if not file_path.exists():
                    raise FileNotFoundError(f"Normalization file not found: {file_path}")
                normvals = json.load(open(file_path))
                norm[mod] = (
                    torch.tensor(normvals["mean"]).float(),
                    torch.tensor(normvals["std"]).float(),
                )
            self.norm = norm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        sample = torch.load(file_path)

        for mod in ["s2", "s1_asc", "s1_des"]:
            if mod in sample and self.temporal_dropout > 0:
                data = sample[mod]
                dates = sample.get(f"{mod}_dates", None)
                if dates is not None and len(dates) > 1 and data.shape[0] == len(dates):
                    N = data.shape[0]
                    keep_n = max(1, int(N * (1 - self.temporal_dropout)))
                    keep_idx = torch.randperm(N)[:keep_n]
                    sample[mod] = data[keep_idx]
                    sample[f"{mod}_dates"] = dates[keep_idx]

        if self.norm is not None:
            for mod in self.modalities:
                if mod in sample:
                    data = sample[mod]
                    mean, std = self.norm[mod]
                    if len(data.shape) == 4:  # (T, C, H, W)
                        sample[mod] = (data - mean[None, :, None, None]) / std[None, :, None, None]
                    else:  # (C, H, W)
                        sample[mod] = (data - mean[:, None, None]) / std[:, None, None]

        return self.transform(sample)

