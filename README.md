# Rectifying Convex Mirror Reflections

Sprin 2026 SNU Topics in Computer Graphics (Computational Imaging) course final project.

## Team Members

- Sukhun Yang (sukhuny@snu.ac.kr)
- Seungho Lee (lxxseunghh@snu.ac.kr)

## Overview

This project explores neural rectification of distorted convex mirror images. The repository currently contains two PyTorch pipelines:

- `src/flow`: conditional flow matching for direct image rectification.
- `src/field`: deterministic spatial warping for rectification through a learned sampling field.

Both pipelines use paired distorted/ground-truth images and share a similar U-Net backbone, but they solve the rectification problem differently. The flow pipeline generates rectified RGB images through flow matching, while the field pipeline predicts a 2-channel warp field and applies `grid_sample` to the distorted image.

## Project Structure

```text
.
|-- .python-version
|-- pyproject.toml
|-- README.md
|-- dataset/
|   |-- <id>_input.png
|   `-- <id>_gt.png
|-- results/
`-- src/
    |-- flow/
    |   |-- main.py
    |   |-- evaluate.py
    |   |-- infer_inputs.py
    |   |-- split_dataset.py
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

The code expects paired PNG files. Each distorted input image must have a matching rectified ground-truth image with the same id:

```text
dataset/
|-- 00000_input.png
|-- 00000_gt.png
|-- 00001_input.png
|-- 00001_gt.png
`-- ...
```

For training, use split folders:

```text
dataset/
|-- train/
|   |-- <id>_input.png
|   `-- <id>_gt.png
`-- val/
    |-- <id>_input.png
    `-- <id>_gt.png
```

`src/flow/main.py` trains directly from `dataset/train` and `dataset/val`. `src/field/main.py` uses split folders when present, or creates a deterministic in-memory train/validation split from a flat dataset directory. The evaluation scripts use `val/` when available, or a deterministic validation subset from a flat dataset.

## Flow Pipeline (`src/flow`)

The flow pipeline trains a conditional flow matching model for direct rectification. The flow dataloader flips distorted inputs horizontally, normalizes RGB values to `[-1, 1]`, builds a foreground mask, appends normalized coordinate channels, and returns:

- `condition`: 6 channels, `[distorted_rgb, mask, grid_x, grid_y]`
- `target`: 3-channel rectified ground-truth image

During training, `CFGFlowMatcher` samples random interpolation time `t`, constructs a noisy intermediate image, and trains the U-Net to predict the velocity from noise to the target image. The loss combines velocity MSE, clean image L1 loss, and low-frequency L1 loss. Sampling uses classifier-free guidance (CFG).

### Flow Modules

- `main.py`: train/sample entrypoint. Requires `--train_name`, logs with Weights & Biases, saves raw model state dict checkpoints, and writes comparison images every 10 epochs.
- `evaluate.py`: evaluates a checkpoint with PSNR and SSIM.
- `infer_inputs.py`: runs inference on a flat folder of `*_input.png` files and saves `*_output.png` files.
- `split_dataset.py`: moves flat paired data into deterministic `train/` and `val/` folders.
- `dataloader.py`: loads paired input/ground-truth images and constructs flow conditions.
- `flow_matching_model.py`: implements `CFGFlowMatcher` loss, velocity prediction, CFG, and sampling.
- `unet.py`: U-Net backbone and related model blocks.
- `nn.py`: neural-network utility layers and helpers.
- `fp16_util.py`: mixed precision helper utilities.

### Flow Usage

Run commands from the repository root.

Preview a deterministic dataset split without moving files:

```bash
python src/flow/split_dataset.py --data_dir dataset --dry_run
```

Move paired files into `dataset/train` and `dataset/val`:

```bash
python src/flow/split_dataset.py --data_dir dataset --val_fraction 0.1 --seed 0
```

Train flow matching:

```bash
python src/flow/main.py \
  --mode train \
  --data_dir dataset \
  --out_dir results \
  --epochs 100 \
  --batch_size 4 \
  --lr 5e-5 \
  --cfg_scale 4.0 \
  --train_name flow_baseline
```

Training writes outputs under a timestamped run folder:

```text
results/<YYMMDD_HHMM_run_name>/
|-- model_ep010.pth
|-- sample_ep010.png
|-- model_ep020.pth
|-- sample_ep020.png
`-- ...
```

Sample with a flow checkpoint:

```bash
python src/flow/main.py \
  --mode sample \
  --data_dir dataset \
  --out_dir results \
  --ckpt results/<run>/model_epXXX.pth \
  --cfg_scale 4.0 \
  --train_name flow_sample
```

`src/flow/main.py` currently requires `--train_name` in both train and sample modes. Sample mode loads the first batch from `dataset/val` and saves `inference_cfg_<scale>.png` under a new timestamped output folder.

Evaluate a flow checkpoint:

```bash
python src/flow/evaluate.py \
  --data_dir dataset \
  --ckpt results/<run>/model_epXXX.pth \
  --batch_size 8 \
  --cfg_scale 4.0 \
  --num_steps 50
```

Run inference on input-only images:

```bash
python src/flow/infer_inputs.py \
  --input_dir dataset/visualization \
  --output_dir dataset/visualization \
  --ckpt results/<run>/model_epXXX.pth \
  --limit 4 \
  --batch_size 4 \
  --cfg_scale 4.0 \
  --num_steps 50 \
  --seed 0
```

For each `<id>_input.png`, `infer_inputs.py` saves `<id>_output.png`.

## Field Pipeline (`src/field`)

The field pipeline trains a deterministic spatial warping model. Instead of generating the rectified image directly, the model predicts a 2-channel flow field. `SpatialWarpingModule` adds that field to an identity grid and applies `torch.nn.functional.grid_sample` to warp the distorted RGB image into the rectified view.

The field dataloader normalizes images to `[-1, 1]`, appends normalized coordinate channels, and returns:

- `model_input`: 5 channels, `[distorted_rgb, grid_x, grid_y]`
- `distorted_rgb`: 3-channel distorted image
- `target`: 3-channel rectified ground-truth image

The training loss is reconstruction MSE plus a total-variation-style flow smoothness penalty weighted by `--smooth_weight`.

### Field Modules

- `main.py`: train/sample entrypoint for deterministic spatial warping. Supports checkpoint resume in train mode.
- `evaluate.py`: evaluates a warp checkpoint with PSNR and SSIM.
- `dataloader.py`: loads paired input/ground-truth images and constructs 5-channel model inputs.
- `warp_model.py`: defines `SpatialWarpingModule`, warp application, and loss components.
- `flow_refiner.py`: flow-refinement helper code kept in the repository, but not used by the current deterministic field entrypoints.
- `unet.py`: U-Net backbone and related model blocks.
- `nn.py`: neural-network utility layers and helpers.
- `fp16_util.py`: mixed precision helper utilities.

### Field Usage

Train deterministic spatial warping:

```bash
python src/field/main.py \
  --mode train \
  --data_dir dataset \
  --out_dir results/field \
  --epochs 100 \
  --batch_size 4 \
  --lr 5e-5 \
  --smooth_weight 100 \
  --save_every 10 \
  --train_name field_baseline
```

By default, field training writes to:

```text
results/field/<YYMMDD_HHMM_run_name>/
|-- warp_model_ep10.pth
|-- sample_ep10.png
|-- warp_model_ep20.pth
|-- sample_ep20.png
`-- ...
```

Resume field training from a checkpoint:

```bash
python src/field/main.py \
  --mode train \
  --data_dir dataset \
  --out_dir results/field \
  --epochs 120 \
  --ckpt results/field/<run>/warp_model_ep100.pth
```

Sample with a field checkpoint:

```bash
python src/field/main.py \
  --mode sample \
  --data_dir dataset \
  --out_dir results/field \
  --ckpt results/field/<run>/warp_model_epXXX.pth
```

Sample mode loads the first validation batch and saves `inference_warp.png`.

Evaluate a field checkpoint:

```bash
python src/field/evaluate.py \
  --data_dir dataset \
  --ckpt results/field/<run>/warp_model_epXXX.pth \
  --batch_size 8
```

The evaluation script reports average PSNR and SSIM for warped outputs.

## Dependencies

The project metadata currently says:

- Python version: `3.11` from `.python-version`
- Python requirement: `>=3.11` in `pyproject.toml`
- Declared package dependency: `torchmetrics>=1.9.0`

The source code also imports packages that are not currently listed in `pyproject.toml`:

- `torch`
- `torchvision`
- `Pillow`
- `tqdm`
- `numpy`
- `wandb`

CUDA is used when available. The scripts fall back to Apple MPS or CPU when CUDA is unavailable.
