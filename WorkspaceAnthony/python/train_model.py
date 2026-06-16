#!/usr/bin/env python3

def train_model(model, yaml_path):
    '''Train the YOLO model.

            Parameters:
                model: The YOLO model to be trained.
                yaml_path: Path to the YAML file containing training data information.
                
            Returns: 
                The trained YOLO model.
    '''
    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=1024,
        batch=16,
        device=0,
        single_cls=True,
        rect=True,
        patience=20
    )
    return model
