from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os
import argparse
import torch

from dataloader import MirrorReflectionsDataset
from flow_matching_model import SimpleUNet, CFGFlowMatcher


# 그냥 기본형으로 만들어 놓은 상태, 추후 실제 코드에 맞게 수정 필요!
def main():
    parser = argparse.ArgumentParser(description="Flow Matching for Convex Mirror Rectification")
    parser.add_argument("--mode", type=str, choices=["train", "sample"], required=True, help="Run mode")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Path to dataset")
    parser.add_argument("--out_dir", type=str, default="./results", help="Path to save outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint for sampling")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleUNet().to(device)
    flow_matcher = CFGFlowMatcher(model)

    if args.mode == "train":
        dataset = MirrorReflectionsDataset(args.data_dir)
        if len(dataset) == 0:
            print(f"Error: No data found in {args.data_dir}. Please populate the dataset.")
            return
            
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        
        print(f"Starting training on {device}...")
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
            for condition, x_1 in pbar:
                condition, x_1 = condition.to(device), x_1.to(device)
                
                optimizer.zero_grad()
                loss = flow_matcher.compute_loss(x_1, condition)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_ep{epoch}.pth"))
                model.eval()
                sample_cond = condition[:4]
                samples = flow_matcher.sample(sample_cond)
                samples = (samples * 0.5 + 0.5).clamp(0, 1)
                gt = (x_1[:4] * 0.5 + 0.5).clamp(0, 1)
                distorted = (sample_cond[:, :3] * 0.5 + 0.5).clamp(0, 1)
                comparison = torch.cat([distorted, samples, gt], dim=0)
                save_image(comparison, os.path.join(args.out_dir, f"sample_ep{epoch}.png"), nrow=4)

    elif args.mode == "sample":
        if not args.ckpt or not os.path.exists(args.ckpt):
            print("Error: Please provide a valid --ckpt path for sampling.")
            return
            
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.eval()
        
        dataset = MirrorReflectionsDataset(args.data_dir)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        condition, gt = next(iter(dataloader))
        condition = condition.to(device)
        
        print("Generating samples...")
        samples = flow_matcher.sample(condition, cfg_scale=3.0)
        
        samples = (samples * 0.5 + 0.5).clamp(0, 1)
        distorted = (condition[:, :3] * 0.5 + 0.5).clamp(0, 1)
        
        comparison = torch.cat([distorted, samples], dim=0)
        save_image(comparison, os.path.join(args.out_dir, "inference_result.png"), nrow=4)
        print(f"Saved inference results to {args.out_dir}/inference_result.png")

if __name__ == "__main__":
    main()