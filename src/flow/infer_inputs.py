import argparse
from collections.abc import Mapping
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from flow_matching_model import CFGFlowMatcher
from unet import UNetModel


def build_model(device):
    return UNetModel(
        image_size=256,
        in_channels=9,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0.1,
        channel_mult=(1, 2, 2, 4, 4),
        use_checkpoint=False,
    ).to(device)


def extract_model_state(checkpoint):
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
    return checkpoint


def build_condition(input_path, image_size, device):
    normalized_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    raw_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    dist_img = Image.open(input_path).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
    dist_tensor = normalized_transform(dist_img)
    raw_dist_tensor = raw_transform(dist_img)
    mask_tensor = (raw_dist_tensor.sum(dim=0, keepdim=True) > 0.05).float()

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, image_size),
        torch.linspace(-1, 1, image_size),
        indexing="ij",
    )
    grid_coords = torch.stack([grid_x, grid_y], dim=0)
    condition = torch.cat([dist_tensor, mask_tensor, grid_coords], dim=0)
    return condition.to(device)


def select_input_files(input_dir, limit):
    input_files = sorted(input_dir.glob("*_input.png"))
    if not input_files:
        raise FileNotFoundError(f"No *_input.png files found in {input_dir}")
    if limit > 0:
        input_files = input_files[:limit]
    return input_files


def output_path_for(input_path, output_dir):
    return output_dir / input_path.name.replace("_input.png", "_output.png")


def main():
    parser = argparse.ArgumentParser(description="Run inference on flat *_input.png image folders.")
    parser.add_argument("--input_dir", type=Path, default=Path("./dataset/visualization"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4, help="Number of input images to run; 0 means all")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = select_input_files(args.input_dir, args.limit)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = build_model(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(extract_model_state(checkpoint))
    model.eval()
    flow_matcher = CFGFlowMatcher(model)

    print(f"Running inference on {len(input_files)} input image(s) from {args.input_dir}")
    with torch.no_grad():
        for start in range(0, len(input_files), args.batch_size):
            batch_files = input_files[start : start + args.batch_size]
            condition = torch.stack(
                [build_condition(input_path, args.image_size, device) for input_path in batch_files]
            )
            samples = flow_matcher.sample(
                condition,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
            )
            samples = (samples * 0.5 + 0.5).clamp(0, 1).cpu()

            for input_path, sample in zip(batch_files, samples):
                output_path = output_path_for(input_path, output_dir)
                save_image(sample, output_path)
                print(f"Saved {output_path}")


if __name__ == "__main__":
    main()