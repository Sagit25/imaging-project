import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataloader import MirrorReflectionsDataset
from warping_model import SpatialWarpingModule
from unet import UNetModel

def main():
    parser = argparse.ArgumentParser(description="Deterministic Spatial Warping for Convex Mirror Rectification")
    parser.add_argument("--mode", type=str, choices=["train", "sample"], required=True)
    parser.add_argument("--data_dir", type=str, default="../dataset")
    parser.add_argument("--out_dir", type=str, default="../results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--smooth_weight", type=float, default=0.05, help="Weight for flow smoothness regularization")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
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
    
    warper = SpatialWarpingModule(model, smoothness_weight=args.smooth_weight)

    if args.mode == "train":
        dataset = MirrorReflectionsDataset(args.data_dir, image_size=256, is_train=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler()
        
        print(f"Starting Spatial Warping training on {device}...")
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
            for model_input, dist_tensor, gt_tensor in pbar:
                model_input = model_input.to(device)
                dist_tensor = dist_tensor.to(device)
                gt_tensor = gt_tensor.to(device)
                
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type):
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
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.eval()
        
        dataset = MirrorReflectionsDataset(args.data_dir, image_size=256, is_train=False)
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