import os
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import wandb

from dataloader import MirrorReflectionsDataset
from flow_matching_model import CFGFlowMatcher
from unet import UNetModel

def main():
    parser = argparse.ArgumentParser(description="Advanced Flow Matching for Convex Mirror Rectification")
    parser.add_argument("--mode", type=str, choices=["train", "sample"], required=True)
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Dataset root; uses <data_dir>/train and <data_dir>/val")
    parser.add_argument("--out_dir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="Scale to balance realism and fidelity")
    parser.add_argument("--train_name", type=str, required=True, help="Name of the training run; used as subfolder under out_dir")
    args = parser.parse_args()

    args.train_name = f"{datetime.now().strftime('%y%m%d_%H%M')}_{args.train_name}"

    out_dir = os.path.join(args.out_dir, args.train_name)
    os.makedirs(out_dir, exist_ok=True)
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
        use_checkpoint=False             
    ).to(device)
    
    flow_matcher = CFGFlowMatcher(model, cfg_drop_rate=0.15)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    if args.mode == "train":
        dataset = MirrorReflectionsDataset(train_dir, image_size=256)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler()

        val_dataset = MirrorReflectionsDataset(val_dir, image_size=256, is_train=False)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        val_indices = torch.randperm(len(val_dataset))[:4].tolist()
        val_samples = [val_dataset[i] for i in val_indices]
        fixed_cond = torch.stack([s[0] for s in val_samples]).to(device)
        fixed_gt = torch.stack([s[1] for s in val_samples]).to(device)

        wandb.init(
            entity="seungho-sukhun",
            project="imaging-project",
            name=args.train_name,
            config=vars(args),
        )
        wandb.watch(model, log="all", log_freq=100)

        print(f"Starting advanced Flow Matching training on {device}...")
        global_step = 0
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
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                global_step += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                wandb.log({"train/loss": loss.item(), "epoch": epoch}, step=global_step)

            avg_epoch_loss = epoch_loss / len(dataloader)

            # Validation pass. Seed deterministically so the sampled t / noise are
            # identical every epoch, making val/loss comparable across epochs; restore
            # the RNG state afterwards so training randomness is unaffected.
            model.eval()
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(0)
            val_loss = 0
            with torch.no_grad():
                for condition, x_1 in val_dataloader:
                    condition, x_1 = condition.to(device), x_1.to(device)
                    with torch.amp.autocast(device_type=device.type):
                        val_loss += flow_matcher.compute_loss(x_1, condition).item()
            avg_val_loss = val_loss / len(val_dataloader)
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

            wandb.log(
                {"train/epoch_loss": avg_epoch_loss, "val/loss": avg_val_loss, "epoch": epoch},
                step=global_step,
            )

            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(out_dir, f"model_ep{epoch:03d}.pth"))
                model.eval()
                with torch.no_grad():
                    samples = flow_matcher.sample(fixed_cond, num_steps=50, cfg_scale=args.cfg_scale)

                samples = (samples * 0.5 + 0.5).clamp(0, 1)
                gt = (fixed_gt * 0.5 + 0.5).clamp(0, 1)
                distorted = (fixed_cond[:, :3] * 0.5 + 0.5).clamp(0, 1)

                comparison = torch.cat([distorted, samples, gt], dim=0)
                save_image(comparison, os.path.join(out_dir, f"sample_ep{epoch:03d}.png"), nrow=4)
                grid = make_grid(comparison, nrow=4)
                wandb.log(
                    {"val/samples": wandb.Image(grid, caption="rows: distorted / generated / ground-truth")},
                    step=global_step,
                )

        wandb.finish()

    elif args.mode == "sample":
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.eval()

        dataset = MirrorReflectionsDataset(val_dir, image_size=256, is_train=False)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        condition, gt = next(iter(dataloader))
        condition = condition.to(device)

        print(f"Generating unwarped samples with optimized CFG Scale ({args.cfg_scale})...")
        samples = flow_matcher.sample(condition, cfg_scale=args.cfg_scale)

        samples = (samples * 0.5 + 0.5).clamp(0, 1)
        distorted = (condition[:, :3] * 0.5 + 0.5).clamp(0, 1)

        comparison = torch.cat([distorted, samples], dim=0)
        save_image(comparison, os.path.join(out_dir, f"inference_cfg_{args.cfg_scale}.png"), nrow=4)

if __name__ == "__main__":
    main()