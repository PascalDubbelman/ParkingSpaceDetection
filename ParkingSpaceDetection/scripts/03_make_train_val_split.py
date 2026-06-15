#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import random
import shutil


DEFAULT_IMAGES = Path("GeoreferenceTest/202606121032215360267/png")
DEFAULT_LABELS = Path("GeoreferenceTest/202606121032215360267/obb_labels")
DEFAULT_DATASET = Path("GeoreferenceTest/202606121032215360267/dataset")


def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_pairs(pairs, dataset_dir, split):
    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for image_path, label_path in pairs:
        shutil.copy2(image_path, image_dir / image_path.name)
        shutil.copy2(label_path, label_dir / label_path.name)


def write_yaml(dataset_dir):
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                "  0: parking_space",
                "",
            ]
        )
    )
    return yaml_path


def main():
    parser = ArgumentParser(description="Build a YOLO OBB train/val dataset from PNG images and OBB txt labels.")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.dataset.exists() and not args.overwrite:
        raise FileExistsError(f"{args.dataset} exists; pass --overwrite to rebuild it")

    pairs = []
    for image_path in sorted(args.images.glob("*.png")):
        label_path = args.labels / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            print(f"missing label, skipped: {image_path.name}")

    if not pairs:
        raise RuntimeError("No PNG/TXT pairs found")

    random.seed(args.seed)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * args.train_ratio)
    if len(pairs) > 1:
        split_idx = min(max(split_idx, 1), len(pairs) - 1)

    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    reset_dir(args.dataset)
    copy_pairs(train_pairs, args.dataset, "train")
    copy_pairs(val_pairs, args.dataset, "val")
    yaml_path = write_yaml(args.dataset)

    print(f"pairs: {len(pairs)}")
    print(f"train: {len(train_pairs)}")
    print(f"val: {len(val_pairs)}")
    print(f"yaml: {yaml_path}")


if __name__ == "__main__":
    main()
