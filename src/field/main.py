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
from flow_refiner import CFGFlowRefiner, build_refiner_condition
from warp_model import SpatialWarpingModule
from unet import UNetModel

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_warp_model(device):
    return UNetModel(
        image_size=256,
        in_channels=5,
        model_channels=128,
        out_channels=2,
        num_res_blocks=2,
        attention_resolutions=(4, 8, 16),
        dropout=0.1,
        channel_mult=(1, 2, 2, 4, 4),
        use_checkpoint=True,
    ).to(device)


def build_refiner_model(device):
    return UNetModel(
        image_size=256,
        in_channels=11,
        model_channels=64,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(4, 8, 16),
        dropout=0.1,
        channel_mult=(1, 2, 2, 4),
        use_checkpoint=True,
    ).to(device)


def build_datasets(data_dir, image_size=256, val_fraction=0.1, seed=0):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    train_dataset = MirrorReflectionsDataset(train_dir, image_size=image_size, is_train=True)
    val_dataset = MirrorReflectionsDataset(val_dir, image_size=image_size, is_train=False)

    if len(train_dataset) > 0 and len(val_dataset) > 0:
        return train_dataset, val_dataset, f"Using split dataset: {train_dir}, {val_dir}"

    flat_dataset = MirrorReflectionsDataset(data_dir, image_size=image_size, is_train=True)
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
    return (
        train_subset,
        val_subset,
        f"Using deterministic flat dataset split from {data_dir}: {train_size} train / {val_size} val",
    )


def autocast_context(device, enabled):
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def wandb_log(run, data, step=None):
    if run is not None:
        wandb.log(data, step=step)


def load_hybrid_checkpoint(path, warp_model, refiner_model, device):
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "warp" not in checkpoint or "refiner" not in checkpoint:
        raise ValueError("--ckpt must point to a hybrid field checkpoint with 'warp' and 'refiner' states")

    warp_model.load_state_dict(checkpoint["warp"])
    refiner_model.load_state_dict(checkpoint["refiner"])
    return checkpoint


def save_hybrid_checkpoint(path, warp_model, refiner_model, optimizer, epoch, args):
    torch.save(
        {
            "epoch": epoch,
            "warp": warp_model.state_dict(),
            "refiner": refiner_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def build_comparison_condition(model_input, dist_tensor, warp_outputs):
    return build_refiner_condition(model_input, dist_tensor, warp_outputs)


def denormalize_image(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def should_use_refiner(sample_stage, epoch, warmup_epochs):
    if sample_stage == "warp":
        return False
    if sample_stage == "refine":
        return True
    return epoch > warmup_epochs


def compute_hybrid_loss(
    warper,
    flow_refiner,
    model_input,
    dist_tensor,
    gt_tensor,
    epoch,
    args,
):
    warp_loss, _, _, warp_outputs = warper.compute_loss(model_input, dist_tensor, gt_tensor)

    if epoch <= args.warp_warmup_epochs:
        refine_loss = torch.zeros((), device=gt_tensor.device)
        loss = warp_loss
    else:
        condition = build_comparison_condition(model_input, dist_tensor, warp_outputs)
        refine_loss = flow_refiner.compute_loss(gt_tensor, condition)
        loss = refine_loss + args.warp_loss_weight * warp_loss

    return loss, warp_loss, refine_loss, warp_outputs


def save_and_log_samples(
    out_dir,
    filename,
    warper,
    flow_refiner,
    model_input,
    dist_tensor,
    gt_tensor,
    args,
    use_refiner,
    wandb_run=None,
    step=None,
):
    warp_outputs = warper.build_warp_outputs(model_input, dist_tensor)
    if use_refiner:
        condition = build_comparison_condition(model_input, dist_tensor, warp_outputs)
        final_pred = flow_refiner.sample(
            condition,
            num_steps=args.sample_steps,
            cfg_scale=args.cfg_scale,
        )
        stage = "refine"
    else:
        final_pred = warp_outputs["warped_rgb"]
        stage = "warp"

    comparison = torch.cat(
        [
            denormalize_image(dist_tensor),
            denormalize_image(warp_outputs["warped_rgb"]),
            denormalize_image(final_pred),
            denormalize_image(gt_tensor),
        ],
        dim=0,
    )
    save_image(comparison, os.path.join(out_dir, filename), nrow=dist_tensor.shape[0])

    if wandb_run is not None:
        grid = make_grid(comparison, nrow=dist_tensor.shape[0])
        wandb_log(
            wandb_run,
            {
                "val/samples": wandb.Image(grid, caption=f"rows: distorted / warped / {stage} / ground-truth"),
                "sample/stage": stage,
            },
            step=step,
        )


def main():
    parser = argparse.ArgumentParser(description="Hybrid Spatial Warping + Flow Matching Refinement")
    parser.add_argument("--mode", type=str, choices=["train", "sample"], required=True)
    parser.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "dataset"))
    parser.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "results" / "field"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4, help="Warp model learning rate")
    parser.add_argument("--refiner_lr", type=float, default=1e-4, help="Flow refiner learning rate")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--smooth_weight", type=float, default=0.05, help="Weight for flow smoothness regularization")
    parser.add_argument("--warp_warmup_epochs", type=int, default=20)
    parser.add_argument("--warp_loss_weight", type=float, default=0.2)
    parser.add_argument("--cfg_drop_rate", type=float, default=0.15)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_stage", type=str, choices=["auto", "warp", "refine"], default="auto")
    parser.add_argument("--save_every", type=int, default=10)
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

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    use_amp = device.type == "cuda"
    if args.sample_steps < 1:
        raise ValueError("--sample_steps must be at least 1")

    warp_model = build_warp_model(device)
    refiner_model = build_refiner_model(device)
    warper = SpatialWarpingModule(warp_model, smoothness_weight=args.smooth_weight)
    flow_refiner = CFGFlowRefiner(refiner_model, cfg_drop_rate=args.cfg_drop_rate)

    train_dataset, val_dataset, dataset_message = build_datasets(
        args.data_dir,
        image_size=256,
    )
    print(dataset_message)

    if args.mode == "train":
        dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )
        if len(dataloader) == 0:
            raise ValueError("No training batches available; lower --batch_size or add more data.")

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if len(val_dataloader) == 0:
            raise ValueError("No validation batches available; lower --batch_size or add more data.")

        optimizer = torch.optim.AdamW(
            [
                {"params": warp_model.parameters(), "lr": args.lr},
                {"params": refiner_model.parameters(), "lr": args.refiner_lr},
            ],
            weight_decay=1e-4,
        )
        if args.ckpt:
            load_hybrid_checkpoint(args.ckpt, warp_model, refiner_model, device)

        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        val_indices = torch.randperm(len(val_dataset))[:4].tolist()
        val_samples = [val_dataset[i] for i in val_indices]
        fixed_input = torch.stack([s[0] for s in val_samples]).to(device)
        fixed_dist = torch.stack([s[1] for s in val_samples]).to(device)
        fixed_gt = torch.stack([s[2] for s in val_samples]).to(device)

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
            wandb.watch(warp_model, log="all", log_freq=100)
            wandb.watch(refiner_model, log="all", log_freq=100)

        print(f"Starting hybrid field training on {device}...")
        global_step = 0
        for epoch in range(1, args.epochs + 1):
            warp_model.train()
            refiner_model.train()
            epoch_loss = 0.0
            epoch_warp_loss = 0.0
            epoch_refine_loss = 0.0
            processed_batches = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
            for batch_idx, (model_input, dist_tensor, gt_tensor) in enumerate(pbar):
                if args.max_batches and batch_idx >= args.max_batches:
                    break

                model_input = model_input.to(device)
                dist_tensor = dist_tensor.to(device)
                gt_tensor = gt_tensor.to(device)

                optimizer.zero_grad()
                with autocast_context(device, use_amp):
                    loss, warp_loss, refine_loss, _ = compute_hybrid_loss(
                        warper,
                        flow_refiner,
                        model_input,
                        dist_tensor,
                        gt_tensor,
                        epoch,
                        args,
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(warp_model.parameters()) + list(refiner_model.parameters()),
                    1.0,
                )
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                epoch_warp_loss += warp_loss.item()
                epoch_refine_loss += refine_loss.item()
                processed_batches += 1
                global_step += 1

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "warp": f"{warp_loss.item():.4f}",
                        "refine": f"{refine_loss.item():.4f}",
                    }
                )
                wandb_log(
                    wandb_run,
                    {
                        "train/loss": loss.item(),
                        "train/warp_loss": warp_loss.item(),
                        "train/refine_loss": refine_loss.item(),
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            if processed_batches == 0:
                raise ValueError("No training batches processed; lower --max_batches or check the dataloader.")

            avg_epoch_loss = epoch_loss / processed_batches
            avg_epoch_warp_loss = epoch_warp_loss / processed_batches
            avg_epoch_refine_loss = epoch_refine_loss / processed_batches

            warp_model.eval()
            refiner_model.eval()
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(0)
            val_loss = 0.0
            val_warp_loss = 0.0
            val_refine_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch_idx, (model_input, dist_tensor, gt_tensor) in enumerate(val_dataloader):
                    if args.max_batches and batch_idx >= args.max_batches:
                        break
                    model_input = model_input.to(device)
                    dist_tensor = dist_tensor.to(device)
                    gt_tensor = gt_tensor.to(device)
                    with autocast_context(device, use_amp):
                        loss, warp_loss, refine_loss, _ = compute_hybrid_loss(
                            warper,
                            flow_refiner,
                            model_input,
                            dist_tensor,
                            gt_tensor,
                            epoch,
                            args,
                        )
                    val_loss += loss.item()
                    val_warp_loss += warp_loss.item()
                    val_refine_loss += refine_loss.item()
                    val_batches += 1
            if val_batches == 0:
                raise ValueError("No validation batches processed; lower --max_batches or check the dataloader.")
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

            avg_val_loss = val_loss / val_batches
            avg_val_warp_loss = val_warp_loss / val_batches
            avg_val_refine_loss = val_refine_loss / val_batches
            wandb_log(
                wandb_run,
                {
                    "train/epoch_loss": avg_epoch_loss,
                    "train/epoch_warp_loss": avg_epoch_warp_loss,
                    "train/epoch_refine_loss": avg_epoch_refine_loss,
                    "val/loss": avg_val_loss,
                    "val/warp_loss": avg_val_warp_loss,
                    "val/refine_loss": avg_val_refine_loss,
                    "epoch": epoch,
                },
                step=global_step,
            )
            print(
                f"Epoch {epoch}: "
                f"avg_loss={avg_epoch_loss:.4f}, "
                f"val_loss={avg_val_loss:.4f}"
            )

            should_save = (args.save_every > 0 and epoch % args.save_every == 0) or (epoch == args.epochs)
            if should_save:
                ckpt_path = os.path.join(out_dir, f"hybrid_field_ep{epoch:03d}.pth")
                save_hybrid_checkpoint(ckpt_path, warp_model, refiner_model, optimizer, epoch, args)
                with torch.no_grad():
                    use_refiner = should_use_refiner(args.sample_stage, epoch, args.warp_warmup_epochs)
                    save_and_log_samples(
                        out_dir,
                        f"sample_ep{epoch:03d}.png",
                        warper,
                        flow_refiner,
                        fixed_input,
                        fixed_dist,
                        fixed_gt,
                        args,
                        use_refiner=use_refiner,
                        wandb_run=wandb_run,
                        step=global_step,
                    )

        if wandb_run is not None:
            wandb.finish()

    elif args.mode == "sample":
        if not args.ckpt:
            raise ValueError("--ckpt is required in sample mode")
        checkpoint = load_hybrid_checkpoint(args.ckpt, warp_model, refiner_model, device)
        warp_model.eval()
        refiner_model.eval()

        dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        model_input, dist_tensor, gt_tensor = next(iter(dataloader))
        model_input = model_input.to(device)
        dist_tensor = dist_tensor.to(device)
        gt_tensor = gt_tensor.to(device)

        print("Generating hybrid field samples...")
        with torch.no_grad():
            checkpoint_epoch = checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0
            checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
            warmup_epochs = checkpoint_args.get("warp_warmup_epochs", args.warp_warmup_epochs)
            use_refiner = should_use_refiner(args.sample_stage, checkpoint_epoch, warmup_epochs)
            save_and_log_samples(
                out_dir,
                "inference_hybrid_field.png",
                warper,
                flow_refiner,
                model_input,
                dist_tensor,
                gt_tensor,
                args,
                use_refiner=use_refiner,
            )


if __name__ == "__main__":
    main()
