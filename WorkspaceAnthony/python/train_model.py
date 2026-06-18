#!/usr/bin/env python3
from pathlib import Path
import shutil

def train_model(model, yaml_path, save_dir):
    '''Train the YOLO model.

            Parameters:
                model: The YOLO model to be trained.
                yaml_path: Path to the YAML file containing training data information.
                save_dir: Directory where the trained model will be saved.
                
            Returns: 
                The trained YOLO model.
    '''
    model.train(
        data=yaml_path,
        epochs=200,
        imgsz=1024,
        batch=16,
        device=0,
        single_cls=True,
        rect=True,
        patience=20,
        project=save_dir
    )

    # YOLO saves everything here:
    run_dir = Path(save_dir) / "yolo_training"

    # Where you want to save PNGs + CSV
    export_dir = Path(save_dir) / "exports"
    export_dir.mkdir(exist_ok=True)

    # Copy PNGs
    for png in run_dir.glob("*.png"):
        shutil.copy(png, export_dir)

    # Copy CSVs
    for csv in run_dir.glob("*.csv"):
        shutil.copy(csv, export_dir)

    return model
