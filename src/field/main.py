import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataloader import MirrorReflectionsDataset
from warp_model import SpatialWarpingModule
from unet import UNetModel

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


def autocast_context(device, enabled):
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def main():
    parser = argparse.ArgumentParser(description="Deterministic Spatial Warping for Convex Mirror Rectification")
    parser.add_argument("--mode", type=str, choices=["train", "sample"], required=True)
    parser.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "dataset"))
    parser.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "results" / "field"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--smooth_weight", type=float, default=0.05, help="Weight for flow smoothness regularization")
    parser.add_argument("--max_batches", type=int, default=0, help="Optional limit for smoke tests; 0 uses all batches")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    use_amp = device.type == "cuda"
    model = build_model(device)
    
    warper = SpatialWarpingModule(model, smoothness_weight=args.smooth_weight)

    if args.mode == "train":
        dataset = MirrorReflectionsDataset(args.data_dir, image_size=256, is_train=True)
        if len(dataset) == 0:
            raise ValueError(f"No input images found in {args.data_dir}")
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        if len(dataloader) == 0:
            raise ValueError("No training batches available; lower --batch_size or add more data.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        
        print(f"Starting Spatial Warping training on {device}...")
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
            for batch_idx, (model_input, dist_tensor, gt_tensor) in enumerate(pbar):
                if args.max_batches and batch_idx >= args.max_batches:
                    break
                model_input = model_input.to(device)
                dist_tensor = dist_tensor.to(device)
                gt_tensor = gt_tensor.to(device)
                
                optimizer.zero_grad()
                with autocast_context(device, use_amp):
                    loss, unwarped_pred, _ = warper.compute_loss(model_input, dist_tensor, gt_tensor)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(args.out_dir, f"warp_model_ep{epoch}.pth"))
                model.eval()
                with torch.no_grad():
                    distorted = (dist_tensor[:4] * 0.5 + 0.5).clamp(0, 1)
                    samples = (unwarped_pred[:4] * 0.5 + 0.5).clamp(0, 1)
                    gt = (gt_tensor[:4] * 0.5 + 0.5).clamp(0, 1)
                    
                    comparison = torch.cat([distorted, samples, gt], dim=0)
                    save_image(comparison, os.path.join(args.out_dir, f"sample_ep{epoch}.png"), nrow=4)

    elif args.mode == "sample":
        if not args.ckpt:
            raise ValueError("--ckpt is required in sample mode")
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.eval()
        
        dataset = MirrorReflectionsDataset(args.data_dir, image_size=256, is_train=False)
        if len(dataset) == 0:
            raise ValueError(f"No input images found in {args.data_dir}")
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        model_input, dist_tensor, gt_tensor = next(iter(dataloader))
        model_input, dist_tensor = model_input.to(device), dist_tensor.to(device)
        
        print("Generating unwarped samples using deterministic grid mapping...")
        with torch.no_grad():
            dummy_t = torch.zeros(model_input.shape[0], device=device)
            flow_field = model(model_input, dummy_t)
            unwarped_pred = warper.forward_warp(dist_tensor, flow_field)
        
        distorted = (dist_tensor * 0.5 + 0.5).clamp(0, 1)
        samples = (unwarped_pred * 0.5 + 0.5).clamp(0, 1)
        gt = (gt_tensor * 0.5 + 0.5).clamp(0, 1)
        
        comparison = torch.cat([distorted, samples, gt], dim=0)
        save_image(comparison, os.path.join(args.out_dir, f"inference_warp.png"), nrow=4)

if __name__ == "__main__":
    main()
