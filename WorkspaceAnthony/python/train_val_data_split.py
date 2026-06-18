#!/usr/bin/env python3
from pathlib import Path
import random
import shutil

def create_dirs(train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
    '''Create directories for training and validation images and labels.

            Parameters:
                a train_img_dir (str): Directory to save the training images.
                b val_img_dir (str): Directory to save the validation images.
                c train_lbl_dir (str): Directory to save the training labels.
                d val_lbl_dir (str): Directory to save the validation labels.
                
            Returns: 
                None
    '''
    # Create folders
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

def split_data(img_dir: Path):
    '''Split data into training and validation sets.

            Parameters:
                a img_dir (str): Directory containing the images to be split.   

            Returns:
                train_imgs and val_imgs: Image paths for training and validation sets.
    '''
    # Collect images
    images = list(img_dir.glob("*.png"))

    # Random seed
    random.seed(42)

    # Random shuffle
    random.shuffle(images)

    # Split data in training (80%) and validation (20%)
    split_idx = int(0.8 * len(images))
    train_imgs = images[:split_idx]
    val_imgs   = images[split_idx:]

    return train_imgs, val_imgs


def move_pairs(img_list, lbl_dir: Path, target_img_dir: Path, target_lbl_dir: Path):
    '''Move image and label pairs to the target directories.

            Parameters:
                a img_list (list): List of image paths to be moved.
                b lbl_dir (str): Directory containing the labels corresponding to the images.
                c target_img_dir (str): Directory to move the images to.
                d target_lbl_dir (str): Directory to move the labels to.

            Returns:
                None
    '''
    # Move images and labels
    for img_path in img_list:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"

        shutil.copy(img_path, target_img_dir / img_path.name)

        if lbl_path.exists():
            shutil.copy(lbl_path, target_lbl_dir / lbl_path.name)
        else:
            print(f" Warning: Label not found for {img_path.name}")
