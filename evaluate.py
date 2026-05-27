import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from dataloader import MirrorReflectionsDataset
from flow_matching_model import FlowUNet, CFGFlowMatcher

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate Convex Mirror Rectification")
    parser.add_argument("--data_dir", type=str, default="./dataset/test", help="Path to test dataset")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FlowUNet().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    
    flow_matcher = CFGFlowMatcher(model)
    
    dataset = MirrorReflectionsDataset(args.data_dir, image_size=256, is_train=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    print(f"Starting evaluation on {len(dataset)} images...")
    
    with torch.no_grad():
        for condition, gt in tqdm(dataloader, desc="Evaluating"):
            condition, gt = condition.to(device), gt.to(device)
            
            samples = flow_matcher.sample(condition, num_steps=50, cfg_scale=4.0)

            samples = (samples * 0.5 + 0.5).clamp(0, 1)
            gt = (gt * 0.5 + 0.5).clamp(0, 1)
            
            psnr_val = psnr_metric(samples, gt)
            ssim_val = ssim_metric(samples, gt)
            
            total_psnr += psnr_val.item()
            total_ssim += ssim_val.item()
            num_batches += 1
            
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    print("-" * 30)
    print(f"Evaluation Results:")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    evaluate()