#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import random
import shutil


def create_yolo_folders(dataset_dir):
    '''Create the standard YOLO train and validation folders.

            Parameters:
                dataset_dir: Main dataset output directory.

            Returns:
                Four paths: train images, val images, train labels, val labels.
    '''
    train_img_dir = dataset_dir / "images" / "train"
    val_img_dir = dataset_dir / "images" / "val"
    train_lbl_dir = dataset_dir / "labels" / "train"
    val_lbl_dir = dataset_dir / "labels" / "val"

    for folder in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    return train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir


def count_labels(label_path):
    '''Count non-empty rows in a YOLO label file.

            Parameters:
                label_path: Path to a YOLO TXT label file.

            Returns:
                Number of non-empty label rows.
    '''
    return sum(1 for line in label_path.read_text().splitlines() if line.strip())


def collect_image_label_pairs(image_dir, label_dir, min_labels_per_tile):
    '''Match PNG images with same-name TXT label files.

            Parameters:
                image_dir: Directory containing PNG images.
                label_dir: Directory containing YOLO TXT labels.
                min_labels_per_tile: Minimum number of labels required.

            Returns:
                List of image and label path pairs.
    '''
    pairs = []
    skipped_missing = 0
    skipped_low_label = 0

    for image_path in sorted(image_dir.glob("*.png")):
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            skipped_missing += 1
            print(f"missing label, skipped: {image_path.name}")
            continue

        if count_labels(label_path) < min_labels_per_tile:
            skipped_low_label += 1
            continue

        pairs.append((image_path, label_path))

    if not pairs:
        raise RuntimeError("No PNG/TXT pairs found")

    print(f"missing labels skipped: {skipped_missing}")
    print(f"low-label tiles skipped: {skipped_low_label}")

    return pairs


def split_pairs_randomly(pairs, train_ratio, seed):
    '''Randomly split image-label pairs into training and validation.

            Parameters:
                pairs: List of image and label path pairs.
                train_ratio: Fraction used for training.
                seed: Random seed for reproducible split.

            Returns:
                Training pairs and validation pairs.
    '''
    random.seed(seed)
    random.shuffle(pairs)

    split_index = int(len(pairs) * train_ratio)

    if len(pairs) > 1:
        split_index = min(max(split_index, 1), len(pairs) - 1)

    train_pairs = pairs[:split_index]
    val_pairs = pairs[split_index:]

    return train_pairs, val_pairs


def copy_pairs(pairs, target_img_dir, target_lbl_dir):
    '''Copy image and label pairs into target YOLO folders.

            Parameters:
                pairs: List of image and label path pairs.
                target_img_dir: Output image folder.
                target_lbl_dir: Output label folder.

            Returns:
                None.
    '''
    for image_path, label_path in pairs:
        shutil.copy2(image_path, target_img_dir / image_path.name)
        shutil.copy2(label_path, target_lbl_dir / label_path.name)


def write_data_yaml(dataset_dir):
    '''Write YOLO data.yaml for one parking_space class.

            Parameters:
                dataset_dir: Main dataset output directory.

            Returns:
                Path to data.yaml.
    '''
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


def reset_dataset_folder(dataset_dir, overwrite):
    '''Delete old dataset folder only when overwrite is allowed.

            Parameters:
                dataset_dir: Main dataset output directory.
                overwrite: Whether existing folder may be replaced.

            Returns:
                None.
    '''
    if dataset_dir.exists() and not overwrite:
        raise FileExistsError(f"{dataset_dir} exists; pass --overwrite to rebuild it")

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    dataset_dir.mkdir(parents=True)


def main():
    parser = ArgumentParser(description="Create a random YOLO train/validation split.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-labels-per-tile", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")
    if args.min_labels_per_tile < 0:
        raise ValueError("--min-labels-per-tile must be >= 0")

    pairs = collect_image_label_pairs(args.images, args.labels, args.min_labels_per_tile)
    train_pairs, val_pairs = split_pairs_randomly(pairs, args.train_ratio, args.seed)

    reset_dataset_folder(args.dataset, args.overwrite)
    train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir = create_yolo_folders(args.dataset)

    copy_pairs(train_pairs, train_img_dir, train_lbl_dir)
    copy_pairs(val_pairs, val_img_dir, val_lbl_dir)
    yaml_path = write_data_yaml(args.dataset)

    print(f"pairs: {len(pairs)}")
    print(f"train: {len(train_pairs)}")
    print(f"val: {len(val_pairs)}")
    print(f"yaml: {yaml_path}")


if __name__ == "__main__":
    main()
