import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataloader import MirrorReflectionsDataset
from flow_refiner import CFGFlowRefiner, build_refiner_condition, build_refiner_weight
from warp_model import SpatialWarpingModule
from unet import UNetModel

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_warp_model(device):
    return UNetModel(
        image_size=256,
        in_channels=7,
        model_channels=128,
        out_channels=2,
        num_res_blocks=2,
        attention_resolutions=(4, 8, 16),
        dropout=0.1,
        channel_mult=(1, 2, 2, 4, 4),
        use_checkpoint=True
    ).to(device)


def build_refiner_model(device):
    return UNetModel(
        image_size=256,
        in_channels=14,
        model_channels=64,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(4, 8, 16),
        dropout=0.1,
        channel_mult=(1, 2, 2, 4),
        use_checkpoint=True
    ).to(device)


def autocast_context(device, enabled):
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


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


def main():
    parser = argparse.ArgumentParser(description="Hybrid Spatial Warping + Flow Matching Refinement")
    parser.add_argument("--mode", type=str, choices=["train", "sample"], required=True)
    parser.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "dataset"))
    parser.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "results" / "field"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4, help="Warp model learning rate")
    parser.add_argument("--refiner_lr", type=float, default=1e-4, help="Flow refiner learning rate")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--smooth_weight", type=float, default=0.05, help="Weight for flow smoothness regularization")
    parser.add_argument("--warp_warmup_epochs", type=int, default=20)
    parser.add_argument("--warp_loss_weight", type=float, default=0.2)
    parser.add_argument("--cfg_drop_rate", type=float, default=0.15)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--border_width", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--max_batches", type=int, default=0, help="Optional limit for smoke tests; 0 uses all batches")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    use_amp = device.type == "cuda"
    if args.sample_steps < 1:
        raise ValueError("--sample_steps must be at least 1")

    warp_model = build_warp_model(device)
    refiner_model = build_refiner_model(device)
    warper = SpatialWarpingModule(warp_model, smoothness_weight=args.smooth_weight)
    flow_refiner = CFGFlowRefiner(refiner_model, cfg_drop_rate=args.cfg_drop_rate)

    if args.mode == "train":
        dataset = MirrorReflectionsDataset(args.data_dir, image_size=256, is_train=True, border_width=args.border_width)
        if len(dataset) == 0:
            raise ValueError(f"No input images found in {args.data_dir}")
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        if len(dataloader) == 0:
            raise ValueError("No training batches available; lower --batch_size or add more data.")

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

        print(f"Starting hybrid field training on {device}...")
        for epoch in range(1, args.epochs + 1):
            warp_model.train()
            refiner_model.train()
            epoch_loss = 0.0
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
                    warp_loss, _, _, warp_outputs = warper.compute_loss(model_input, dist_tensor, gt_tensor)

                    if epoch <= args.warp_warmup_epochs:
                        loss = warp_loss
                        refine_loss = torch.zeros((), device=device)
                    else:
                        condition = build_comparison_condition(model_input, dist_tensor, warp_outputs)
                        pixel_weight = build_refiner_weight(warp_outputs)
                        refine_loss = flow_refiner.compute_loss(gt_tensor, condition, pixel_weight=pixel_weight)
                        loss = refine_loss + args.warp_loss_weight * warp_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(warp_model.parameters()) + list(refiner_model.parameters()),
                    1.0,
                )
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                processed_batches += 1
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "warp": f"{warp_loss.item():.4f}",
                        "refine": f"{refine_loss.item():.4f}",
                    }
                )

            if processed_batches == 0:
                raise ValueError("No training batches processed; lower --max_batches or check the dataloader.")

            should_save = (args.save_every > 0 and epoch % args.save_every == 0) or (epoch == args.epochs)
            if should_save:
                ckpt_path = os.path.join(args.out_dir, f"hybrid_field_ep{epoch:03d}.pth")
                save_hybrid_checkpoint(ckpt_path, warp_model, refiner_model, optimizer, epoch, args)

                warp_model.eval()
                refiner_model.eval()
                with torch.no_grad():
                    sample_count = min(4, dist_tensor.shape[0])
                    sample_input = model_input[:sample_count]
                    sample_dist = dist_tensor[:sample_count]
                    sample_gt = gt_tensor[:sample_count]
                    sample_outputs = warper.build_warp_outputs(sample_input, sample_dist)
                    sample_condition = build_comparison_condition(sample_input, sample_dist, sample_outputs)
                    sample_refined = flow_refiner.sample(
                        sample_condition,
                        num_steps=args.sample_steps,
                        cfg_scale=args.cfg_scale,
                    )

                    distorted = (sample_dist * 0.5 + 0.5).clamp(0, 1)
                    warped = (sample_outputs["warped_rgb"] * 0.5 + 0.5).clamp(0, 1)
                    refined = (sample_refined * 0.5 + 0.5).clamp(0, 1)
                    gt = (sample_gt * 0.5 + 0.5).clamp(0, 1)
                    comparison = torch.cat([distorted, warped, refined, gt], dim=0)
                    save_image(comparison, os.path.join(args.out_dir, f"sample_ep{epoch:03d}.png"), nrow=sample_count)

            avg_epoch_loss = epoch_loss / processed_batches
            print(f"Epoch {epoch}: avg_loss={avg_epoch_loss:.4f}")

    elif args.mode == "sample":
        if not args.ckpt:
            raise ValueError("--ckpt is required in sample mode")
        load_hybrid_checkpoint(args.ckpt, warp_model, refiner_model, device)
        warp_model.eval()
        refiner_model.eval()

        dataset = MirrorReflectionsDataset(args.data_dir, image_size=256, is_train=False, border_width=args.border_width)
        if len(dataset) == 0:
            raise ValueError(f"No input images found in {args.data_dir}")
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        model_input, dist_tensor, gt_tensor = next(iter(dataloader))
        model_input = model_input.to(device)
        dist_tensor = dist_tensor.to(device)
        gt_tensor = gt_tensor.to(device)

        print("Generating hybrid field samples...")
        with torch.no_grad():
            warp_outputs = warper.build_warp_outputs(model_input, dist_tensor)
            condition = build_comparison_condition(model_input, dist_tensor, warp_outputs)
            refined_pred = flow_refiner.sample(condition, num_steps=args.sample_steps, cfg_scale=args.cfg_scale)

        distorted = (dist_tensor * 0.5 + 0.5).clamp(0, 1)
        warped = (warp_outputs["warped_rgb"] * 0.5 + 0.5).clamp(0, 1)
        refined = (refined_pred * 0.5 + 0.5).clamp(0, 1)
        gt = (gt_tensor * 0.5 + 0.5).clamp(0, 1)

        comparison = torch.cat([distorted, warped, refined, gt], dim=0)
        save_image(comparison, os.path.join(args.out_dir, "inference_hybrid_field.png"), nrow=dist_tensor.shape[0])


if __name__ == "__main__":
    main()
