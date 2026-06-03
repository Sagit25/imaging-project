import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from dataloader import MirrorReflectionsDataset
from warping_model import SpatialWarpingModule
from unet import UNetModel

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate Deterministic Spatial Warping Rectification")
    parser.add_argument("--data_dir", type=str, default="./dataset/test", help="Path to test dataset")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNetModel(
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
    
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    
    warper = SpatialWarpingModule(model)
    
    dataset = MirrorReflectionsDataset(args.data_dir, image_size=256, is_train=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    print(f"Starting evaluation on {len(dataset)} images...")
    
    with torch.no_grad():
        for model_input, dist_tensor, gt_tensor in tqdm(dataloader, desc="Evaluating"):
            model_input = model_input.to(device)
            dist_tensor = dist_tensor.to(device)
            gt_tensor = gt_tensor.to(device)
            
            dummy_t = torch.zeros(model_input.shape[0], device=device)
            flow_field = model(model_input, dummy_t)
            unwarped_pred = warper.forward_warp(dist_tensor, flow_field)

            samples = (unwarped_pred * 0.5 + 0.5).clamp(0, 1)
            gt = (gt_tensor * 0.5 + 0.5).clamp(0, 1)
            
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