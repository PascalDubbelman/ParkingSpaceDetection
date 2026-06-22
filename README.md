# Remote Sensing and GIS Integration (GRS60312) Project (ACT\_RGIC08).

## The Increasing Pressure on Parking Spots

There are many challenges related to car parking spaces in urban areas. One of the challenges is to provide a simpler and cheaper alternative for parking inventories. This project addresses the potential of deep learning to provide a reliable and scalable method for automatically detecting individual parking spaces in Wageningen.



### Objective

The main objective of this study is to evaluate the feasibility of using deep learning algorithms and open-source high-resolution aerial imagery to automatically detect individual parking spaces in Wageningen.



### Repository Usage

The scrips are made to run as a pipeline. In order to run the scripts the directory paths in the beginning of the notebooks need to be changed to fit the users repository.

This repository consist of three folders:
* Create\_Tiles -Creating tiles from aerial imagery to fit the YOLO models.
** NPCd0\_SplitTiles.ipynb -Split tiles into 1024x1024 pixels with overlap
** NPCd1\_RenameTiles.ipynb -Give appropriate naming to the tiles
** NPCd2\_RandomSplitTestImages.ipynb -Randomly split the images into test tiles

*ObjectDetection\_with\_OrientatedBB -Scripts to run the model (YoloV11) with orientated bounding boxes.

*ObjectDetection\_with\_RegularBB -Scripts to run the models (Yolov11 and YoloV9) with regular bounding boxes.
**data.yaml -YAML file to define how layers connect
**mainYoloV9.ipynb -Pipeline to run YOLO V9.
**mainYoloV11.ipynb -Pipeline to run YOLO v11.
**'Python' folder
***georeference\_images.py -Script to create georeferenced images.
***tif\_to\_png.py -Script to convert .tif to .png file.
***train\_model.py -Script to train the YOLO model.
***train\_val\_data\_split.py -Script to split training and validation data.
***xml\_to\_txt.py -Script to convert .xml to .txt file.

### Contributions

