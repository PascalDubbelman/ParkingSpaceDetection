#!/usr/bin/env python3
from pathlib import Path
import rasterio
from PIL import Image


def tif_to_png(tif_dir, png_dir):
    '''Convert all .tif files in tif_dir to .png and save them in png_dir.

            Parameters:
                a tif_dir (str): Directory containing .tif files.
                b png_dir (str): Directory to save the converted .png files.
                
            Returns: 
                None
    '''
    png_dir.mkdir(parents=True, exist_ok=True)
    for tif_path in tif_dir.glob("*.tif"):
        with rasterio.open(tif_path) as src:
            img = src.read([1, 2, 3])
            img = img.transpose(1, 2, 0)
            out_path = png_dir / (tif_path.stem + ".png")
            Image.fromarray(img).save(out_path)
            