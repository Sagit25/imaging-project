import os
import argparse
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

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


def build_datasets(data_dir, image_size=256, val_fraction=0.1, seed=0):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    train_dataset = MirrorReflectionsDataset(train_dir, image_size=image_size)
    val_dataset = MirrorReflectionsDataset(val_dir, image_size=image_size, is_train=False)

    if len(train_dataset) > 0 and len(val_dataset) > 0:
        return train_dataset, val_dataset, f"Using split dataset: {train_dir}, {val_dir}"

    flat_dataset = MirrorReflectionsDataset(data_dir, image_size=image_size)
    if len(flat_dataset) == 0:
        raise ValueError(
            f"No input images found in {data_dir}. Expected *_input.png files in "
            f"{data_dir} or split folders {train_dir} and {val_dir}."
        )

    if len(flat_dataset) == 1:
        return flat_dataset, flat_dataset, f"Using the single image in {data_dir} for both train and val"

    val_size = max(1, int(round(len(flat_dataset) * val_fraction)))
    train_size = len(flat_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(flat_dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset, f"Using deterministic flat dataset split from {data_dir}: {train_size} train / {val_size} val"


def autocast_context(device, enabled):
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def wandb_log(run, data, step=None):
    if run is not None:
        wandb.log(data, step=step)


def main():
    parser = argparse.ArgumentParser(description="Advanced Flow Matching for Convex Mirror Rectification")
    parser.add_argument("--mode", type=str, choices=["train", "sample"], required=True)
    parser.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "dataset"), help="Dataset root; uses <data_dir>/train and <data_dir>/val when present, otherwise splits a flat dataset")
    parser.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "results" / "flow"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="Scale to balance realism and fidelity")
    parser.add_argument("--sample_steps", type=int, default=50, help="Number of ODE steps used when sampling")
    parser.add_argument("--max_batches", type=int, default=0, help="Optional limit for smoke tests; 0 uses all batches")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--train_name", type=str, default="", help="Name of the run; used as subfolder under out_dir")
    parser.add_argument("--wandb_mode", type=str, choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb_project", type=str, default="imaging-project")
    parser.add_argument("--wandb_entity", type=str, default="seungho-sukhun")
    args = parser.parse_args()

    if args.mode == "train":
        run_label = args.train_name or "run"
        args.train_name = f"{datetime.now().strftime('%y%m%d_%H%M')}_{run_label}"
    else:
        args.train_name = args.train_name or (Path(args.ckpt).stem if args.ckpt else "sample")

    out_dir = os.path.join(args.out_dir, args.train_name)
    os.makedirs(out_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    use_amp = device.type == "cuda"
    
    model = build_model(device)
    flow_matcher = CFGFlowMatcher(model, cfg_drop_rate=0.15)
    train_dataset, val_dataset, dataset_message = build_datasets(args.data_dir, image_size=256)
    print(dataset_message)

    if args.mode == "train":
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        if len(dataloader) == 0:
            raise ValueError("No training batches available; lower --batch_size or add more data.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        val_indices = torch.randperm(len(val_dataset))[:4].tolist()
        val_samples = [val_dataset[i] for i in val_indices]
        fixed_cond = torch.stack([s[0] for s in val_samples]).to(device)
        fixed_gt = torch.stack([s[1] for s in val_samples]).to(device)

        wandb_run = None
        if args.wandb_mode != "disabled":
            if wandb is None:
                raise ImportError("wandb is not installed; use --wandb_mode disabled or install wandb.")
            wandb_run = wandb.init(
                entity=args.wandb_entity or None,
                project=args.wandb_project,
                name=args.train_name,
                config=vars(args),
                mode=args.wandb_mode,
            )
            wandb.watch(model, log="all", log_freq=100)

        print(f"Starting advanced Flow Matching training on {device}...")
        global_step = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0
            processed_batches = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
            for batch_idx, (condition, x_1) in enumerate(pbar):
                if args.max_batches and batch_idx >= args.max_batches:
                    break
                condition, x_1 = condition.to(device), x_1.to(device)

                optimizer.zero_grad()
                with autocast_context(device, use_amp):
                    loss = flow_matcher.compute_loss(x_1, condition)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                processed_batches += 1
                global_step += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                wandb_log(wandb_run, {"train/loss": loss.item(), "epoch": epoch}, step=global_step)

            avg_epoch_loss = epoch_loss / processed_batches

            # Validation pass. Seed deterministically so the sampled t / noise are
            # identical every epoch, making val/loss comparable across epochs; restore
            # the RNG state afterwards so training randomness is unaffected.
            model.eval()
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(0)
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for batch_idx, (condition, x_1) in enumerate(val_dataloader):
                    if args.max_batches and batch_idx >= args.max_batches:
                        break
                    condition, x_1 = condition.to(device), x_1.to(device)
                    with autocast_context(device, use_amp):
                        val_loss += flow_matcher.compute_loss(x_1, condition).item()
                    val_batches += 1
            avg_val_loss = val_loss / val_batches
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

            wandb_log(
                wandb_run,
                {"train/epoch_loss": avg_epoch_loss, "val/loss": avg_val_loss, "epoch": epoch},
                step=global_step,
            )

            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(out_dir, f"model_ep{epoch:03d}.pth"))
                model.eval()
                with torch.no_grad():
                    samples = flow_matcher.sample(fixed_cond, num_steps=args.sample_steps, cfg_scale=args.cfg_scale)

                samples = (samples * 0.5 + 0.5).clamp(0, 1)
                gt = (fixed_gt * 0.5 + 0.5).clamp(0, 1)
                distorted = (fixed_cond[:, :3] * 0.5 + 0.5).clamp(0, 1)

                comparison = torch.cat([distorted, samples, gt], dim=0)
                save_image(comparison, os.path.join(out_dir, f"sample_ep{epoch:03d}.png"), nrow=4)
                if wandb_run is not None:
                    grid = make_grid(comparison, nrow=4)
                    wandb_log(
                        wandb_run,
                        {"val/samples": wandb.Image(grid, caption="rows: distorted / generated / ground-truth")},
                        step=global_step,
                    )

        if wandb_run is not None:
            wandb.finish()

    elif args.mode == "sample":
        if not args.ckpt:
            raise ValueError("--ckpt is required in sample mode")
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.eval()

        dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

        condition, gt = next(iter(dataloader))
        condition = condition.to(device)

        print(f"Generating unwarped samples with optimized CFG Scale ({args.cfg_scale})...")
        samples = flow_matcher.sample(condition, num_steps=args.sample_steps, cfg_scale=args.cfg_scale)

        samples = (samples * 0.5 + 0.5).clamp(0, 1)
        distorted = (condition[:, :3] * 0.5 + 0.5).clamp(0, 1)

        comparison = torch.cat([distorted, samples], dim=0)
        save_image(comparison, os.path.join(out_dir, f"inference_cfg_{args.cfg_scale}.png"), nrow=4)

if __name__ == "__main__":
    main()
