# Remote Sensing and GIS Integration (GRS60312) Project (ACT_RGIC08)

## The Increasing Pressure on Parking Spots

Urban areas face increasing challenges related to car parking spaces. One key challenge is finding simpler and more cost‑effective alternatives for conducting parking inventories.  
This project explores the potential of deep learning to provide a reliable and scalable method for automatically detecting individual parking spaces in Wageningen.

## Objective

The main objective of this study is to evaluate the feasibility of using deep learning algorithms and open‑source high‑resolution aerial imagery to automatically detect individual parking spaces in Wageningen.

## Repository Usage

The scripts in this repository are designed to run as a pipeline.  
Before running them, update the directory paths at the beginning of each notebook to match your local repository structure.

This repository consists of 2 main folders:

---

## 1. `Create_Tiles`
Tools for generating image tiles suitable for YOLO models.

- `NPCd0_SplitTiles.ipynb` — Split aerial imagery into 1024×1024 px tiles with overlap  
- `NPCd1_RenameTiles.ipynb` — Assign standardized names to tiles  
- `NPCd2_RandomSplitTestImages.ipynb` — Randomly split images into test tiles  

---

## 2. `Combine_Geojsons`

This step is **optional**, but required if your annotation export produces **multiple `.geojson` files**.  
Before running `NPCd5_YoloV11_OBB_Pipeline.ipynb`, you must merge all exported annotation files into a single GeoJSON.

Create an input folder inside `Combine_Geojsons` and place all exported `.geojson` files there.

From the **root of the ParkingSpaceDetection repository**, run:

```bash
python Combine_Geojsons/NPCd3_CombineGeojson.py \
  --input Combine_Geojsons/input_geojsons \
  --output Combine_Geojsons/combined_annotations.geojson```

- `NPCd3_CombineGeojson.py` - Creates a combined `.geojson` file. 

---

## 3. `Object_Detection_Models`
Scripts to run YOLOv11 and YOLOv9 with regular bounding boxes.

- `data.yaml` — Defines dataset structure  
- `NPCd4_YoloV9_HBB_Pipeline.ipynb` — Pipeline for YOLOv9 with horizontal bounding boxes
- `NPCd5_YoloV11_HBB_Pipeline.ipynb` — Pipeline for YOLOv11 with horizontal bounding boxes
- `NPCd6_YoloV11_OBB_Pipeline.ipynb` - Pipeline for YOLOv11 with orientated bounding boxes

### Python utilities for models with horizontal bounding boxes (inside the `PythonHBB` folder)

- `tif_to_png.py` — Convert `.tif` to `.png` 
- `xml_to_txt.py` — Convert `.xml` annotations to `.txt` format   
- `train_val_data_split.py` — Split training and validation datasets  
- `train_model.py` — Train YOLO models 
- `georeference_HBB.py` — Generate georeferenced bounding boxes  

### Python utilities for models with orientated bounding boxes (inside the `PythonOBB` folder)
- `01_tif_to_png.py` — Convert `.tif` to `.png` 
- `02_geojson_to_yolo_obb_no_dedup.py` - Used the `.geotiff` and `.geojson` to produce normalized coordinates for the yolo OBB `.txt` 
- `03_train_val_data_split.py` — Split training and validation datasets  
- `04_train_obb.py` - Train YOLO OBB model
- `05_predict_obb.py` - Predicts parking spaces using trained model
- `06_predictions_to_geojson.py` - Generate georeferenced orientated bounding boxes

---

## Step-by-step guide
### Setup
The notebooks should be run in Google Colab. To be able to do that; upload the repository to your google drive. The step guide begins at round 1, however, begin at round 1 only if you want to completely reproduce our workflow. For best results/models start at round 2 (images with overlap=better for model).  

### Round 1
- Follow tutorial Download data
- Run `NPCd0_SplitTiles.ipynb` to split tiles
- Follow tutorial Sorting tiles
- Follow tutorial Annotation round 1
- Run `NPCd1_RenameTiles.ipynb` to rename tiles
- Run model pipelines `NPCd3_YoloV9_HBB_Pipeline.ipynb` and `NPCd4_YoloV11_HBB_Pipeline.ipynb`

### Round 2
- Follow tutorial Download data
- Run `NPCd0_SplitTiles.ipynb` to split tiles
- Follow tutorial Sorting tiles
- Follow tutorial Field survey
- Follow tutorial Annotation round 2
- Run `NPCd1_RenameTiles.ipynb` to rename tiles
- Run model pipelines `NPCd3_YoloV9_HBB_Pipeline.ipynb`, `NPCd4_YoloV11_HBB_Pipeline.ipynb` and `NPCd5_YoloV11_OBB_Pipeline.ipynb`

### Round test tiles
- Follow tutorial Download data
- Run `NPCd0_SplitTiles.ipynb` to split tiles
- Follow tutorial Sorting tiles
- Run `NPCd2_RandomSplitTestImages.ipynb` to random select images for testing
- Follow tutorial Field survey
- Follow tutorial Annotation round 2
- Run `NPCd1_RenameTiles.ipynb` to rename tiles
- Run model pipelines `NPCd3_YoloV9_HBB_Pipeline.ipynb`, `NPCd4_YoloV11_HBB_Pipeline.ipynb` and `NPCd5_YoloV11_OBB_Pipeline.ipynb`

## Contributions
The authors of this repository are Polly Cheung, Pascal Dubbelman, Anthony Jansen, Iris Lagemaat, and Susanna van de Wetering.

*Last edited: 25/06/2026*


