import os
import argparse
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.path.join(os.environ["XDG_CACHE_HOME"], "fontconfig"), exist_ok=True)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from dataloader import MirrorReflectionsDataset
from flow_matching_model import CFGFlowMatcher
from unet import UNetModel

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_model(device):
    return UNetModel(
        image_size=256,
        in_channels=9,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(4, 8, 16),
        dropout=0.1,
        channel_mult=(1, 2, 2, 4, 4),
        use_checkpoint=False
    ).to(device)


def build_eval_dataset(data_dir):
    val_dir = os.path.join(data_dir, "val")
    val_dataset = MirrorReflectionsDataset(val_dir, image_size=256, is_train=False)
    if len(val_dataset) > 0:
        return val_dataset, f"Using validation dataset: {val_dir}"

    flat_dataset = MirrorReflectionsDataset(data_dir, image_size=256, is_train=False)
    if len(flat_dataset) == 0:
        raise ValueError(
            f"No input images found in {data_dir}. Expected *_input.png files in "
            f"{data_dir} or {val_dir}."
        )
    return flat_dataset, f"Using flat dataset for evaluation: {data_dir}"

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate Convex Mirror Rectification")
    parser.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "dataset"), help="Path to test dataset")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=0, help="Optional limit for smoke tests; 0 evaluates all batches")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    model = build_model(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    
    flow_matcher = CFGFlowMatcher(model)
    
    dataset, dataset_message = build_eval_dataset(args.data_dir)
    print(dataset_message)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    print(f"Starting evaluation on {len(dataset)} images...")
    
    with torch.no_grad():
        for batch_idx, (condition, gt) in enumerate(tqdm(dataloader, desc="Evaluating")):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            condition, gt = condition.to(device), gt.to(device)
            
            samples = flow_matcher.sample(condition, num_steps=args.num_steps, cfg_scale=args.cfg_scale)

            samples = (samples * 0.5 + 0.5).clamp(0, 1)
            gt = (gt * 0.5 + 0.5).clamp(0, 1)
            
            psnr_val = psnr_metric(samples, gt)
            ssim_val = ssim_metric(samples, gt)
            
            total_psnr += psnr_val.item()
            total_ssim += ssim_val.item()
            num_batches += 1
            
    if num_batches == 0:
        raise ValueError("No evaluation batches available; lower --batch_size or add more data.")

    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    print("-" * 30)
    print(f"Evaluation Results:")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    evaluate()
