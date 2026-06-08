import os
import argparse
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.path.join(os.environ["XDG_CACHE_HOME"], "fontconfig"), exist_ok=True)

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from dataloader import MirrorReflectionsDataset
from flow_refiner import CFGFlowRefiner, build_refiner_condition
from main import build_refiner_model, build_warp_model, load_hybrid_checkpoint, should_use_refiner
from warp_model import SpatialWarpingModule

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_model(device):
    return UNetModel(
        image_size=256,
        in_channels=5,
        model_channels=128,
        out_channels=2,
        num_res_blocks=2,
        attention_resolutions=(4, 8, 16),
        dropout=0.1,
        channel_mult=(1, 2, 2, 4, 4),
        use_checkpoint=True
    ).to(device)


def build_eval_dataset(data_dir, image_size=256, val_fraction=0.1, seed=0):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    train_dataset = MirrorReflectionsDataset(train_dir, image_size=image_size)
    val_dataset = MirrorReflectionsDataset(val_dir, image_size=image_size, is_train=False)

    if len(train_dataset) > 0 and len(val_dataset) > 0:
        return val_dataset, f"Using validation dataset: {val_dir}"

    flat_dataset = MirrorReflectionsDataset(data_dir, image_size=image_size)
    if len(flat_dataset) == 0:
        raise ValueError(
            f"No input images found in {data_dir}. Expected *_input.png files in "
            f"{data_dir} or split folders {train_dir} and {val_dir}."
        )

    if len(flat_dataset) == 1:
        return flat_dataset, f"Using the single image in {data_dir} for validation"

    val_size = max(1, int(round(len(flat_dataset) * val_fraction)))
    train_size = len(flat_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    _, val_subset = random_split(flat_dataset, [train_size, val_size], generator=generator)
    return (
        val_subset,
        f"Using deterministic flat dataset validation split from {data_dir}: "
        f"{train_size} train / {val_size} val",
    )


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Spatial Warping + Flow Matching Refinement")
    parser.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "dataset"), help="Path to test dataset")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to hybrid field checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_stage", type=str, choices=["auto", "warp", "refine"], default="auto")
    parser.add_argument("--warp_warmup_epochs", type=int, default=20)
    parser.add_argument("--max_batches", type=int, default=0, help="Optional limit for smoke tests; 0 evaluates all batches")
    args = parser.parse_args()
    if args.sample_steps < 1:
        raise ValueError("--sample_steps must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    model = build_model(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    
    warper = SpatialWarpingModule(model)
    
    dataset, dataset_message = build_eval_dataset(args.data_dir)
    print(dataset_message)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    refined_psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    refined_ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    warped_psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    warped_ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_refined_psnr = 0.0
    total_refined_ssim = 0.0
    total_warped_psnr = 0.0
    total_warped_ssim = 0.0
    num_batches = 0

    print(f"Starting hybrid evaluation on {len(dataset)} images...")

    with torch.no_grad():
        for batch_idx, (model_input, dist_tensor, gt_tensor) in enumerate(tqdm(dataloader, desc="Evaluating")):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            model_input = model_input.to(device)
            dist_tensor = dist_tensor.to(device)
            gt_tensor = gt_tensor.to(device)

            warp_outputs = warper.build_warp_outputs(model_input, dist_tensor)
            warped = (warp_outputs["warped_rgb"] * 0.5 + 0.5).clamp(0, 1)
            gt = (gt_tensor * 0.5 + 0.5).clamp(0, 1)

            if use_refiner:
                condition = build_refiner_condition(model_input, dist_tensor, warp_outputs)
                refined_pred = flow_refiner.sample(condition, num_steps=args.sample_steps, cfg_scale=args.cfg_scale)
                refined = (refined_pred * 0.5 + 0.5).clamp(0, 1)
                total_refined_psnr += refined_psnr_metric(refined, gt).item()
                total_refined_ssim += refined_ssim_metric(refined, gt).item()
            total_warped_psnr += warped_psnr_metric(warped, gt).item()
            total_warped_ssim += warped_ssim_metric(warped, gt).item()
            num_batches += 1

    if num_batches == 0:
        raise ValueError("No evaluation batches available; lower --batch_size or add more data.")

    print("-" * 30)
    print("Evaluation Results:")
    print(f"Warped PSNR:  {total_warped_psnr / num_batches:.4f} dB")
    print(f"Warped SSIM:  {total_warped_ssim / num_batches:.4f}")
    if use_refiner:
        print(f"Refined PSNR: {total_refined_psnr / num_batches:.4f} dB")
        print(f"Refined SSIM: {total_refined_ssim / num_batches:.4f}")
    else:
        print("Refined metrics skipped; sample_stage resolved to warp")
    print("-" * 30)


if __name__ == "__main__":
    evaluate()
