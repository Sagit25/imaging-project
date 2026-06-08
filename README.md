# Computational Imaging (4190.762) Project

SNU Topics in Computer Graphics (Computational Imaging) course final project.

## Project Information

This project explores neural rectification of distorted convex mirror images. 
It contains two PyTorch-based pipelines:

- `src/flow`: a direct conditional flow matching model that maps distorted inputs to rectified target images.
- `src/field`: a hybrid model that first predicts a spatial warp field and then optionally refines the warped result with flow matching.

## Project Structure

```text
.
|-- README.md
|-- dataset/
|   |-- <id>_input.png
|   `-- <id>_gt.png
`-- src/
    |-- flow/
    |   |-- main.py
    |   |-- evaluate.py
    |   |-- dataloader.py
    |   |-- flow_matching_model.py
    |   |-- unet.py
    |   |-- nn.py
    |   `-- fp16_util.py
    `-- field/
        |-- main.py
        |-- evaluate.py
        |-- dataloader.py
        |-- warp_model.py
        |-- flow_refiner.py
        |-- unet.py
        |-- nn.py
        `-- fp16_util.py
```

## Dataset Format

The dataset is expected to contain paired PNG files:

```text
dataset/
|-- 00000_input.png
|-- 00000_gt.png
|-- 00001_input.png
|-- 00001_gt.png
`-- ...
```

Each `*_input.png` file is the distorted convex mirror image, and the matching `*_gt.png` file is the rectified ground-truth image. The loader matches pairs by the shared file id before `_input` or `_gt`.

Both pipelines also support split folders:

```text
dataset/
|-- train/
|   |-- <id>_input.png
|   `-- <id>_gt.png
`-- val/
    |-- <id>_input.png
    `-- <id>_gt.png
```

If `train/` and `val/` are not present, the training scripts create a deterministic train/validation split from the flat dataset directory.

## Pipelines

### Conditional Flow Matching (`src/flow`)

The flow pipeline trains a conditional flow matching model for direct image rectification. The condition tensor contains the distorted RGB image, a foreground mask, and normalized coordinate channels. The U-Net predicts a velocity field that transports noise toward the rectified target image.

Key behavior:

- Uses classifier-free guidance (CFG) during sampling.
- Maintains an exponential moving average (EMA) copy of the model for checkpoints and sampling.
- Saves validation comparison images with distorted input, generated output, and ground truth.

### Spatial Warping (`src/field`)

The field pipeline decomposes the task into two stages:

1. A spatial warping network predicts a 2-channel sampling field and warps the distorted RGB image with `grid_sample`.
2. A flow refiner optionally improves the warped result using conditional flow matching.

Key behavior:

- Trains the warp model first during a warmup period.
- Adds the flow refiner after `--warp_warmup_epochs`.
- Saves comparison images with distorted input, warped result, final prediction, and ground truth.
- Supports `--sample_stage warp`, `--sample_stage refine`, or `--sample_stage auto`.

## Module Guide

### `src/flow`

- `main.py`: training and sampling entrypoint for direct flow matching. It builds datasets, models, optimizers, EMA checkpoints, validation samples, and optional Weights & Biases logging.
- `evaluate.py`: evaluates a flow checkpoint with PSNR and SSIM.
- `dataloader.py`: loads paired `*_input.png` and `*_gt.png` images, flips distorted inputs horizontally for mirror reconstruction, normalizes tensors to `[-1, 1]`, builds masks, and appends coordinate channels.
- `flow_matching_model.py`: implements `CFGFlowMatcher`, including training loss, classifier-free guidance dropout, velocity prediction, and ODE-style sampling.
- `unet.py`: U-Net backbone and related blocks used by the flow model.
- `nn.py`: shared neural network utility layers and functions such as convolution helpers, normalization, timestep embeddings, EMA updates, and gradient checkpointing.
- `fp16_util.py`: mixed precision helper utilities for FP16 training support.

### `src/field`

- `main.py`: training and sampling entrypoint for the hybrid spatial field and flow-refinement pipeline.
- `evaluate.py`: evaluates hybrid checkpoints with warped and, when enabled, refined PSNR/SSIM metrics.
- `dataloader.py`: loads paired images, normalizes them, appends coordinate channels to the model input, and returns tensors for the warper and refiner.
- `warp_model.py`: defines `SpatialWarpingModule`, which predicts sampling grids, applies `torch.nn.functional.grid_sample`, and computes reconstruction plus flow smoothness losses.
- `flow_refiner.py`: implements `CFGFlowRefiner` and builds the refiner condition from warped RGB, distorted RGB, and coordinate channels.
- `unet.py`: U-Net backbone and related blocks used by both the warp model and refiner.
- `nn.py`: shared neural network utility layers and functions.
- `fp16_util.py`: mixed precision helper utilities for FP16 training support.

## Usage

Run commands from the repository root.

### Train Direct Flow Matching

```bash
python src/flow/main.py --mode train
```

Useful options:

```bash
python src/flow/main.py \
  --mode train \
  --data_dir dataset \
  --out_dir results/flow \
  --epochs 100 \
  --batch_size 4 \
  --train_name experiment
```

### Sample Direct Flow Matching

```bash
python src/flow/main.py \
  --mode sample \
  --ckpt results/flow/<run>/model_epXXX.pth
```

### Evaluate Direct Flow Matching

```bash
python src/flow/evaluate.py \
  --ckpt results/flow/<run>/model_epXXX.pth
```

### Train Hybrid Field + Refiner

```bash
python src/field/main.py --mode train
```

Useful options:

```bash
python src/field/main.py \
  --mode train \
  --data_dir dataset \
  --out_dir results/field \
  --epochs 100 \
  --batch_size 4 \
  --warp_warmup_epochs 60 \
  --train_name experiment
```

### Sample Hybrid Field + Refiner

```bash
python src/field/main.py \
  --mode sample \
  --ckpt results/field/<run>/hybrid_field_epXXX.pth
```

To force a stage:

```bash
python src/field/main.py \
  --mode sample \
  --ckpt results/field/<run>/hybrid_field_epXXX.pth \
  --sample_stage refine
```

### Evaluate Hybrid Field + Refiner

```bash
python src/field/evaluate.py \
  --ckpt results/field/<run>/hybrid_field_epXXX.pth
```

## Outputs

Training and sampling outputs are written under `results/` by default:

- `results/flow/<run>/model_epXXX.pth`: direct flow checkpoint.
- `results/flow/<run>/sample_epXXX.png`: flow validation comparison image.
- `results/flow/<run>/inference_cfg_<scale>.png`: flow sampling result.
- `results/field/<run>/hybrid_field_epXXX.pth`: hybrid field checkpoint.
- `results/field/<run>/sample_epXXX.png`: hybrid validation comparison image.
- `results/field/<run>/inference_hybrid_field.png`: hybrid sampling result.

When `--train_name` is provided in training mode, the scripts prepend a timestamp to create a run directory such as `results/flow/260608_1530_experiment`.

## Dependencies

The repository does not currently include a dependency lockfile. The code imports the following main packages:

- Python 3
- PyTorch
- torchvision
- torchmetrics
- Pillow
- tqdm
- numpy
- wandb, optional for experiment logging

CUDA AMP is used when training on CUDA. The scripts also support Apple MPS or CPU fallback when CUDA is unavailable.
