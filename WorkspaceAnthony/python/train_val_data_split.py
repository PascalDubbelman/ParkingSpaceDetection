#!/usr/bin/env python3
from pathlib import Path 
import shutil

def create_dirs(train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
    # Create folders
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

def split_data(img_dir: Path):
   
    # Images in numerical order
    images = sorted(img_dir.glob("*.png"), key=lambda p: int(p.stem))

    # Split data in training (80%) and validation (20%)
    split_idx = int(0.8 * len(images))
    train_imgs = images[:split_idx]
    val_imgs   = images[split_idx:]

    return train_imgs, val_imgs


def move_pairs(img_list, lbl_dir: Path, target_img_dir: Path, target_lbl_dir: Path):
    for img_path in img_list:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"

        shutil.copy(img_path, target_img_dir / img_path.name)

        if lbl_path.exists():
            shutil.copy(lbl_path, target_lbl_dir / lbl_path.name)
        else:
            print(f" Warning: Label not found for {img_path.name}")
