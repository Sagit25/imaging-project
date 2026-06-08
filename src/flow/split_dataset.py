import argparse
import shutil
from pathlib import Path

import torch

DEFAULT_VAL_FRACTION = 0.1
DEFAULT_SEED = 0


def find_input_files(data_dir):
    return sorted(data_dir.glob("*_input.png"))


def matching_gt_path(input_path):
    file_id = input_path.name.rsplit("_input", 1)[0]
    return input_path.with_name(f"{file_id}_gt.png")


def deterministic_split(input_files, val_fraction, seed):
    if not input_files:
        raise ValueError("No *_input.png files found.")

    if len(input_files) == 1:
        return [], input_files

    val_size = max(1, int(round(len(input_files) * val_fraction)))
    train_size = len(input_files) - val_size
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(input_files), generator=generator).tolist()

    train_inputs = [input_files[idx] for idx in indices[:train_size]]
    val_inputs = [input_files[idx] for idx in indices[train_size:]]
    return train_inputs, val_inputs


def validate_pairs(input_files, train_dir, val_dir):
    missing = []
    collisions = []

    for dest_dir, files in ((train_dir, input_files["train"]), (val_dir, input_files["val"])):
        for input_path in files:
            gt_path = matching_gt_path(input_path)
            if not gt_path.is_file():
                missing.append(str(gt_path))

            for src_path in (input_path, gt_path):
                dest_path = dest_dir / src_path.name
                if dest_path.exists():
                    collisions.append(str(dest_path))

    if missing:
        raise FileNotFoundError("Missing ground-truth files:\n" + "\n".join(missing))
    if collisions:
        raise FileExistsError("Destination files already exist:\n" + "\n".join(collisions))


def move_pair(input_path, dest_dir):
    gt_path = matching_gt_path(input_path)
    shutil.move(str(input_path), str(dest_dir / input_path.name))
    shutil.move(str(gt_path), str(dest_dir / gt_path.name))


def main():
    parser = argparse.ArgumentParser(
        description="Move a flat dataset into deterministic train/val folders."
    )
    parser.add_argument("--data_dir", type=Path, default=Path("./dataset"))
    parser.add_argument("--val_fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry_run", action="store_true", help="Print split sizes without moving files")
    args = parser.parse_args()

    if not 0 < args.val_fraction < 1:
        raise ValueError("--val_fraction must be between 0 and 1")

    data_dir = args.data_dir
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    input_files = find_input_files(data_dir)
    train_inputs, val_inputs = deterministic_split(input_files, args.val_fraction, args.seed)
    split_inputs = {"train": train_inputs, "val": val_inputs}

    validate_pairs(split_inputs, train_dir, val_dir)

    print(
        f"Deterministic split from {data_dir}: "
        f"{len(train_inputs)} train / {len(val_inputs)} val "
        f"(seed={args.seed}, val_fraction={args.val_fraction})"
    )

    if args.dry_run:
        print("Dry run only; no files moved.")
        return

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for input_path in train_inputs:
        move_pair(input_path, train_dir)
    for input_path in val_inputs:
        move_pair(input_path, val_dir)

    print("Done moving dataset pairs.")


if __name__ == "__main__":
    main()