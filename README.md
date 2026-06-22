# Remote Sensing and GIS Integration (GRS60312) Project (ACT_RGIC08)

## The Increasing Pressure on Parking Spots

Urban areas face increasing challenges related to car parking spaces. One key challenge is finding simpler and more cost‑effective alternatives for conducting parking inventories.  
This project explores the potential of deep learning to provide a reliable and scalable method for automatically detecting individual parking spaces in Wageningen.

## Objective

The main objective of this study is to evaluate the feasibility of using deep learning algorithms and open‑source high‑resolution aerial imagery to automatically detect individual parking spaces in Wageningen.

## Repository Usage

The scripts in this repository are designed to run as a pipeline.  
Before running them, update the directory paths at the beginning of each notebook to match your local repository structure.

This repository consists of three main folders:

---

## 1. `Create_Tiles`
Tools for generating image tiles suitable for YOLO models.

- **NPCd0_SplitTiles.ipynb** — Split aerial imagery into 1024×1024 px tiles with overlap  
- **NPCd1_RenameTiles.ipynb** — Assign standardized names to tiles  
- **NPCd2_RandomSplitTestImages.ipynb** — Randomly split images into test tiles  

---

## 2. `ObjectDetection_with_OrientatedBB`
Scripts to run YOLOv11 with oriented bounding boxes.

---

## 3. `ObjectDetection_with_RegularBB`
Scripts to run YOLOv11 and YOLOv9 with regular bounding boxes.

- **data.yaml** — Defines dataset structure  
- **mainYoloV9.ipynb** — Pipeline for YOLOv9  
- **mainYoloV11.ipynb** — Pipeline for YOLOv11  

### Python utilities (inside the `Python` folder)

- **georeference_images.py** — Generate georeferenced images  
- **tif_to_png.py** — Convert `.tif` to `.png`  
- **train_model.py** — Train YOLO models  
- **train_val_data_split.py** — Split training and validation datasets  
- **xml_to_txt.py** — Convert `.xml` annotations to `.txt` format  

---

## Contributions



