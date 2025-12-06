# PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models

This documentation present the modifications made to PANGAEA Benchmark including:

- **RAMEN** implementation
- **FLOPs and inference time** computation using [fvcore](https://github.com/facebookresearch/fvcore) library.

# Setup

Clone the repository
Clone the repository:
```
git clone https://github.com/nicolashoudre/RAMEN
cd RAMEN/pangaea-bench
```

**Dependencies**

We provide several ways to install the dependencies.

1. **Using either Conda or Mamba**:
    ```
    conda env create -f environment.yaml
    conda activate pangaea-bench
    ```

    Optional: install [Mamba](https://github.com/conda-forge/miniforge/releases/) for faster resolution times
    ```
    wget https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Mambaforge-24.3.0-0-Linux-x86_64.sh
    sh ./Mambaforge-24.3.0-0-Linux-x86_64.sh

    mamba env create -f environment.yaml
    mamba activate pangaea-bench
    ```

2. **Using pip**, create a Python native virtual environment and install dependencies into it:
   ```
   export PANGAEA_PATH=/path/to/venv/pangaea-bench # change this
   python3 -m venv ${PANGAEA_PATH}
   source ${PANGAEA_PATH}/bin/activate
   
   pip install -r requirements.txt
   ```
 **Then install the code repository as a development package**
   ```
   pip install --no-build-isolation --no-deps -e .
   ```

# Training with RAMEN

RAMEN encoder is integrated seamlessly in PANGAEA framework. Due to the resolution-adjustable nature of our model, training configuration has to be tuned per dataset and wanted target GSD. 

Specifically, Input GSD, Input size and target GSD have to be specified.

We provide in this section corresponding training scripts for all evaluated datasets in our paper:

## HLS BurnScars
```
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=hlsburnscars \
   encoder=ramen \
   encoder.input_res=30.0 \
   encoder.input_size=512 \ # To tune 
   encoder.res=480.0 \ # To tune 
   decoder=seg_upernet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation
```

## MADOS
```
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=mados \
   encoder=ramen \
   encoder.input_res=10.0 \
   encoder.input_size=120 \ # To tune 
   encoder.res=40.0 \ # To tune 
   decoder=seg_upernet \
   preprocessing=seg_focus_crop \
   criterion=cross_entropy \
   task=segmentation
```

## Pastis

Multi-temporal model setup:
```
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=pastis \
   encoder=ramen \
   encoder.input_res=10.0 \
   encoder.input_size=128 \ # To tune 
   encoder.res=40.0 \ # To tune 
   decoder=seg_upernet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation
```

Mono-temporal model with late LTAE fusion setup:
```
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=pastis \
   encoder=ramen_monotemporal \
   encoder.input_res=10.0 \
   encoder.input_size=128 \ # To tune 
   encoder.res=40.0 \ # To tune 
   decoder=seg_upernet_mt_ltae \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation
```

## Sen1Floods11
```
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=sen1floods11 \
   encoder=ramen \
   encoder.input_res=10.0 \
   encoder.input_size=128 \ # To tune 
   encoder.res=40.0 \ # To tune 
   decoder=seg_upernet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation
```

## DynamicEarthNet
Multi-temporal model setup:
```
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=dynamicen\
   encoder=ramen \
   encoder.input_res=3.0 \
   encoder.input_size=256 \ # To tune 
   encoder.res=24.0 \ # To tune 
   decoder=seg_upernet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation
```

Mono-temporal model with late LTAE fusion setup:
```
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=dynamicen \
   encoder=ramen_monotemporal \
   encoder.input_res=3.0 \
   encoder.input_size=256 \ # To tune 
   encoder.res=24.0 \ # To tune 
   decoder=seg_upernet_mt_ltae \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation
```

## CropTypeMapping - South Sudan
Multi-temporal model setup:
```
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=croptypemapping\
   encoder=ramen \
   encoder.input_res=10.0 \
   encoder.input_size=64 \ # To tune 
   encoder.res=40.0 \ # To tune 
   decoder=seg_upernet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation
```

Mono-temporal model with late LTAE fusion setup:
```
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=croptypemapping \
   encoder=ramen_monotemporal \
   encoder.input_res=10.0 \
   encoder.input_size=64 \ # To tune 
   encoder.res=40.0 \ # To tune 
   decoder=seg_upernet_mt_ltae \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation
```

## SpaceNet7
```
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=spacenet7 \
   encoder=ramen \
   encoder.input_res=40.0 \
   encoder.input_size=128 \ # To tune 
   encoder.res=16.0 \ # To tune 
   decoder=seg_upernet \
   preprocessing=seg_default \
   criterion=dice \
   task=segmentation
```

## AI4SmallFarms

```
 torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
 --config-name=train \
 dataset=ai4smallfarms \
 encoder=ramen \
 decoder=seg_upernet \
 preprocessing=seg_default \
 criterion=dice \
 task=segmentation \
 data_replicate=2 \
 task.trainer.best_metric_key=IoU
```

⚠️ As no validation set is provided for AI4SmallFarms, we evaluated our model on the last trained checkpoint. You can do so by renaming last checkpoint to *checkpoint__best.pt* and running:

```
torchrun pangaea/run.py --config-name=test ckpt_dir=path_to_ckpt_dir
```

# FLOPs and inference time calculation

To assess the compute/performance trade-off of our model, we implemented a **FLOPs and inference time** computation using [fvcore](https://github.com/facebookresearch/fvcore) library.

To evaluate your own configuration, simply run :

```
torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=hlsburnscars \
   encoder=ramen \
   encoder.input_res = 30.0 \
   encoder.input_size = 512 \ # To tune 
   encoder.res = 480.0 \ # To tune 
   decoder=seg_upernet\
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=profiler \
   task.trainer.n_epochs=1 \
   batch_size=1
```

# Citation

If you use PANGAEA, please cite:

```
@misc{marsocci2024pangaeaglobalinclusivebenchmark,
      title={PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models}, 
      author={Valerio Marsocci and Yuru Jia and Georges Le Bellier and David Kerekes and Liang Zeng and Sebastian Hafner and Sebastian Gerard and Eric Brune and Ritu Yadav and Ali Shibli and Heng Fang and Yifang Ban and Maarten Vergauwen and Nicolas Audebert and Andrea Nascetti},
      year={2024},
      eprint={2412.04204},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.04204}, 
}
```

