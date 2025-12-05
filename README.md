# RAMEN: Resolution-Adjustable Multimodal Encoder for Earth Observation

[![arXiv](https://img.shields.io/badge/arXiv-2515.05025-b31b1b.svg)](https://arxiv.org/abs/2512.05025)
[![HuggingFace](https://img.shields.io/badge/-HuggingFace-3B4252?style=flat&logo=huggingface&logoColor=)](https://huggingface.co/nicolashoudre/RAMEN)
[![python](https://img.shields.io/badge/-Python_3.10+-blue?logo=python&logoColor=white)]()
[![pytorch](https://img.shields.io/badge/PyTorch_2.4+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.4+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

Official implementation of [RAMEN: Resolution-Adjustable Multimodal Encoder for Earth Observation](https://arxiv.org/abs/2512.05025).

**News:**

- **05/12/2025**: [ArXiv pre-print](https://arxiv.org/abs/2512.05025) and [encoder weights](https://huggingface.co/nicolashoudre/RAMEN) are available ! 
Hugging face transformer integration and demo notebook will be released soon.

# Abstract

🌍 Earth observation (EO) data spans a wide range of spatial, spectral, and temporal resolutions, from high-resolution optical imagery to low resolution multispectral products or radar time series.

🍜 We introduce **RAMEN**, a resolution-adjustable multimodal encoder that learns a shared visual representation across EO data in a fully sensor-agnostic manner. RAMEN treats the modality and spatial and temporal resolutions as key input data features, enabling coherent analysis across modalities within a unified latent space. Its main methodological contribution is to define spatial resolution as a controllable output parameter, giving users direct control over the desired level of detail at inference and allowing explicit trade-offs between spatial precision and computational cost. 

<p align="center">
  <img src=".figures/Intro_RAMEN.png" alt="RAMEN workflow" width="400"/>
</p>

Direct link to encoder weights is available at [<img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="20"/> HuggingFace](https://huggingface.co/nicolashoudre/RAMEN).

# Key features

- 🛰️ **Sensor-agnostic foundation model**: RAMEN supports any kind of multispectral, SAR or elevation maps modalities. Just specify input shape, channels and original spatial resolution (GSD) !
- 🔧 **Adjustable feature map resolution**: Customize the resolution of feature maps to suit specific downstream tasks and computational constraints.
- 🌍 **Multimodal data fusion**: Effectively combine data from multiple modalities into a unified representation.

<p align="center">
  <img src=".figures/MAE.png" alt="RAMEN workflow" width="800"/>
</p>

# PANGAEA Bench evaluation

All downstream tasks results presented in RAMEN where conducted using the [PANGAEA](https://github.com/VMarsocci/pangaea-bench) Benchmark. We report here the main results obtained on eight tasks.

| Model | BurnSr | MADOS | PASTIS | Sen1Fl11 | DEN | CTM-SS | SN7 | AI4Farms | Avg. mIoU | Avg. Rank |
|-------|---------|--------|--------|----------|------|--------|------|-----------|-----------|-----------|
| CROMA | 82.42 | 67.55 | 32.32 | 90.89 | 38.29 | 49.38 | 59.28 | 25.65 | 55.72 | 6.50 |
| DOFA | 80.63 | 59.58 | 30.02 | 89.37 | 39.29 | 51.33 | **61.84** | 27.07 | 54.89 | 7.50 |
| TerraMind-B | 82.42 | 69.52 | 40.51 | 90.62 | 37.87 | **55.80** | 60.61 | 28.12 | 58.18 | 4.25 |
| TerraMind-L | 82.93 | **75.57** | **43.13** | 90.78 | 37.89 | 55.04 | 59.98 | 27.47 | 59.10 | 3.75 |
| **RAMEN (ours)** | **85.02** | 69.72 | 42.29 | **91.03** | **39.85** | 53.27 | 60.31 | **38.78** | **60.03** | **2.63** |


More informations on how to reproduce results and implement RAMEN in PANGAEA can be found in the [`pangaea-bench`](./pangaea-bench) folder.

# Pretraining

Pretraining was conducted using a masked autoencoding strategy on three multimodal datasets:

- [FLAIR-HUB](https://huggingface.co/datasets/IGNF/FLAIR-HUB):  A multi-sensor land-cover dataset with very high resolution RGB-NIR imagery, 10 bands Sentinel-2 (S2) time series, VV/VH Sentinel-1 (S1) time series and elevation (DSM/DTM) maps covering France.
- [WorldStrat](https://github.com/worldstrat/worldstrat): A global collection covering 10000 $\text{km}^2$ of matched high resolution RGB-NIR imagery and low resolution 12 bands S2 time-series stratified to all type of land-use across the world. 
- [MMEarth](https://github.com/vishalned/MMEarth-data): A large corpus of 1.2 million locations distributed around the world combining 13 bands S2 imagery, 8 bands S1 spanning all available ascending/descending polarizations and elevation (DSM/slope) maps.

More informations on how to pretrain RAMEN can be found in the [`pretraining`](./pretraining) folder.

# Reference 

If you use RAMEN, please cite our paper:

```
@article{RAMEN,
  title={{RAMEN}: Resolution-Adjustable Multimodal Encoder for Earth Observation},
  author={Nicolas Houdré and Diego Marcos and Hugo Riffaud de Turckheim and Dino Ienco and Laurent Wendling and Camille Kurtz and Sylvain Lobry},
  journal={arXiv preprint arXiv:2512.05025},
  year={2025}
}
```

# Acknowledgment

- This project was built using [Lightning-Hydra](https://github.com/ashleve/lightning-hydra-template) template.
- Pretraining framework and multi dataset training implementation comes from [AnySat](https://github.com/gastruc/AnySat).
- Downstream tasks evaluation is conducted using [PANGAEA](https://github.com/VMarsocci/pangaea-bench) Benchmark.

This project was supported by ANR project ANR-23-IAS1-0002 GEO ReSeT and provided with computing AI and storage resources by GENCI at IDRIS thanks to the grant 2025-AD011016746 on the supercomputer Jean Zay's H100 partition.
