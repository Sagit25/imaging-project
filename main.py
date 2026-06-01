import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataloader import MirrorReflectionsDataset
from flow_matching_model import CFGFlowMatcher
from unet import UNetModel

def main():
    parser = argparse.ArgumentParser(description="Advanced Flow Matching for Convex Mirror Rectification")
    parser.add_argument("--mode", type=str, choices=["train", "sample"], required=True)
    parser.add_argument("--data_dir", type=str, default="./dataset")
    parser.add_argument("--out_dir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="Scale to balance realism and fidelity")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Define unet model
    model = UNetModel(
        image_size=256,
        in_channels=9,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),   
        dropout=0.1,
        channel_mult=(1, 2, 2, 4, 4),   
        use_checkpoint=True             
    ).to(device)
    
    flow_matcher = CFGFlowMatcher(model, cfg_drop_rate=0.15)

    if args.mode == "train":
        dataset = MirrorReflectionsDataset(args.data_dir, image_size=256)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler()
        
        print(f"Starting advanced Flow Matching training on {device}...")
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
            for condition, x_1 in pbar:
                condition, x_1 = condition.to(device), x_1.to(device)
                
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type):
                    loss = flow_matcher.compute_loss(x_1, condition)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_ep{epoch}.pth"))
                model.eval()
                sample_cond = condition[:4]
                samples = flow_matcher.sample(sample_cond, num_steps=50, cfg_scale=args.cfg_scale)
                
                samples = (samples * 0.5 + 0.5).clamp(0, 1)
                gt = (x_1[:4] * 0.5 + 0.5).clamp(0, 1)
                distorted = (sample_cond[:, :3] * 0.5 + 0.5).clamp(0, 1)
                
                comparison = torch.cat([distorted, samples, gt], dim=0)
                save_image(comparison, os.path.join(args.out_dir, f"sample_ep{epoch}.png"), nrow=4)

    elif args.mode == "sample":
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.eval()
        
        dataset = MirrorReflectionsDataset(args.data_dir, image_size=256, is_train=False)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        condition, gt = next(iter(dataloader))
        condition = condition.to(device)
        
        print(f"Generating unwarped samples with optimized CFG Scale ({args.cfg_scale})...")
        samples = flow_matcher.sample(condition, cfg_scale=args.cfg_scale)
        
        samples = (samples * 0.5 + 0.5).clamp(0, 1)
        distorted = (condition[:, :3] * 0.5 + 0.5).clamp(0, 1)
        
        comparison = torch.cat([distorted, samples], dim=0)
        save_image(comparison, os.path.join(args.out_dir, f"inference_cfg_{args.cfg_scale}.png"), nrow=4)

if __name__ == "__main__":
    main()