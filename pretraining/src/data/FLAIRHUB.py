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

def dict_datetimes_to_yday(datetime_dict: dict, start: int = 0) -> torch.Tensor:
    """
    Convert a dict of YYYYMMDD strings into a tensor of day-of-year integers.
    """
    dates = []
    for idx in range(start, len(datetime_dict) + start):
        date_str = str(datetime_dict[str(idx)])
        dt = datetime.strptime(date_str, "%Y%m%d")
        dates.append(dt.timetuple().tm_yday)
    return torch.tensor(dates, dtype=torch.int32)

def filter_time_series(
    data_tensor: torch.Tensor,
    max_cloud_value: float = 1.0,
    max_snow_value: float = 1.0,
    max_fraction_covered: float = 0.05,
) -> torch.Tensor:
    """
    Filters time steps based on per-pixel cloud and snow values across an image sequence.
    """
    T, C, H, W = data_tensor.shape
    num_pix = H * W
    threshold = (1 - max_fraction_covered) * num_pix

    select = (data_tensor[:, 1, :, :] <= max_cloud_value) & (data_tensor[:, 0, :, :] <= max_snow_value)
    valid_counts = select.view(T, -1).sum(dim=1)
    selected_idx = valid_counts >= threshold

    if not selected_idx.any():
        snow_valid = (data_tensor[:, 0, :, :] <= max_snow_value).view(T, -1).sum(dim=1)
        selected_idx = snow_valid >= threshold

    return selected_idx


class FLAIRHUB(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        partition: float = 1.0,
        num_classes: int = 19,
        label_choice: str = "cosia",  # 'cosia' or 'lpis'
        norm_path=None,
        temporal_dropout: float = 0.0,
        classif: bool = False,
    ):
        self.path = path
        self.transform = transform
        self.partition = partition
        self.num_classes = num_classes
        self.temporal_dropout = temporal_dropout
        self.classif = classif
        self.split = split
        self.collate_fn = collate_fn

        self.mod_mapping = {
            "aerial": "AERIAL_RGBI",
            "aerial_rlt": "AERIAL-RLT_PAN",
            "dem": "DEM_ELEV",
            "spot": "SPOT_RGBI",
            "s2": "SENTINEL2_TS",
            "s2_mask": "SENTINEL2_MSK-SC",
            "s1_asc": "SENTINEL1-ASC_TS",
            "s1_des": "SENTINEL1-DESC_TS",
            "cosia": "AERIAL_LABEL-COSIA",
            "lpis": "ALL_LABEL-LPIS",
        }

        self.modalities = modalities
        if label_choice not in ["cosia", "lpis"]:
            raise ValueError("label_choice must be 'cosia' or 'lpis'")
        self.label_choice = self.mod_mapping[label_choice]

        csv_file = os.path.join(path, f"{split}.csv")
        df = pd.read_csv(csv_file, sep=";")

        if partition < 1.0:
            self.data = df.sample(frac=partition, random_state=42).reset_index(drop=True)
        else:
            self.data = df
        self.labels = None

        print("initiating_dates_dict")
        self.dates_dict = {}
        mtd_path = os.path.join(path, "GLOBAL_ALL_MTD")
        for mod in self.modalities:
            if mod=="dem" and "aerial" not in self.modalities:
                mod="aerial"
            mod_csv = self.mod_mapping[mod]
            gpkg_file = os.path.join(mtd_path, f"GLOBAL_{mod_csv}_MTD_DATES.gpkg")
            if os.path.exists(gpkg_file):
                gdf = gpd.read_file(gpkg_file, engine="pyogrio", use_arrow=True)
                self.dates_dict[mod_csv] = {
                    row["patch_id"]: row.get("date") or row.get("acquisition_dates")
                    for _, row in gdf.iterrows()
                }
        if "dem" in self.modalities:
            self.dates_dict["DEM_ELEV"] = self.dates_dict["AERIAL_RGBI"]
        print("loaded_dates_dict")
                
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
        for i in range(len(self.data)):
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

    def load_dates(self, mod_csv, patch_id):
        """Return day-of-year dates for given modality and patch."""
        mod_dict = self.dates_dict.get(mod_csv)
        if mod_dict is None:
            raise ValueError(f"No date metadata found for modality {mod_csv}")
        date_entry = mod_dict.get(patch_id)
        if date_entry is None:
            raise KeyError(f"No dates for patch_id {patch_id} in modality {mod_csv}")
        if mod_csv in ["AERIAL_RGBI", "AERIAL-RLT_PAN", "DEM_ELEV", "SPOT_RGBI"]:
            dt = datetime.strptime(date_entry, "%Y%m%d")
            return torch.tensor([dt.timetuple().tm_yday], dtype=torch.int32)
        else:
            return(dict_datetimes_to_yday(json.loads(date_entry), start=1))

    def __getitem__(self, i):
        line = self.data.iloc[i]
        patch_id = line["patch_id"]
        output = {"name": patch_id}

        with rasterio.open((Path(self.path) / line[self.label_choice]).resolve()) as f:
            labels = f.read()[0].astype("int32")
        labels[labels > self.num_classes-2] = self.num_classes-1
        output["label"] = torch.FloatTensor(labels)

        for mod in self.modalities:
            mod_csv = self.mod_mapping[mod]
            path_mod = (Path(self.path) / line[mod_csv]).resolve()
            with rasterio.open(path_mod) as f:
                data = torch.FloatTensor(f.read())

            if mod in ["s2", "s1_asc", "s1_des"]:

                C = 10 if mod == "s2" else 2  # number of bands
                T = data.shape[0] // C
                data = data.view(T, C, data.shape[1], data.shape[2])  # [T, C, H, W]
        
            output[mod] = data
            dates = self.load_dates(mod_csv, patch_id)
            output[f"{mod}_dates"] = dates

            if mod == "s2":
                path_mask = (Path(self.path) / line["SENTINEL2_MSK-SC"]).resolve()
                with rasterio.open(path_mask) as f:
                    mask_data = torch.FloatTensor(f.read())  # [T, 2, H, W] or similar
                T = mask_data.shape[0] // 2
                mask_data = mask_data.view(T, 2, mask_data.shape[1], mask_data.shape[2])  # [T, C, H, W]
                idx_valid = filter_time_series(
                    mask_data,
                    max_cloud_value=1,
                    max_snow_value=1,
                    max_fraction_covered=0.05
                )
                data = data[idx_valid]
                dates = dates[idx_valid]
                output[mod] = data
                output[f"{mod}_dates"] = dates

            if (
                self.split == "train"
                and self.temporal_dropout > 0
                and len(dates) > 1
                and data.shape[0] == len(dates)
            ):
                N = data.shape[0]
                keep_n = max(1, int(N * (1 - self.temporal_dropout)))
                keep_idx = torch.randperm(N)[:keep_n]
                output[mod] = data[keep_idx]
                output[f"{mod}_dates"] = dates[keep_idx]


            if mod in ["aerial", "spot", "dem", "aerial_rlt"]:
                output[mod] = output[mod].unsqueeze(0) # 1, C, H, W to match time series 

        # Normalization
        if self.norm is not None:
            for mod in self.modalities:
                data = output[mod]
                mean, std = self.norm[mod]
                if len(data.shape) == 4:  # (T, C, H, W)
                    output[mod] = (data - mean[None, :, None, None]) / std[None, :, None, None]
                else:  # (C, H, W)
                    output[mod] = (data - mean[:, None, None]) / std[:, None, None]

        return self.transform(output)

    def __len__(self):
        return len(self.data)

class FLAIRHUB_PT(Dataset):
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

