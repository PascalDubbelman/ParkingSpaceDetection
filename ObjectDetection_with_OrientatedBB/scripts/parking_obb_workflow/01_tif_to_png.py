#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import rasterio
from PIL import Image


def convert_one_tif(tif_path, png_dir):
    '''Convert one GeoTIFF image to one PNG image.

            Parameters:
                tif_path: Path to the GeoTIFF file.
                png_dir: Directory where the PNG file should be saved.

            Returns:
                Path to the created PNG file.
    '''
    out_path = png_dir / f"{tif_path.stem}.png"

    with rasterio.open(tif_path) as src:
        if src.count < 3:
            raise ValueError(f"{tif_path.name} has {src.count} band(s), expected at least 3")

        image_array = src.read([1, 2, 3])
        image_array = image_array.transpose(1, 2, 0)

    Image.fromarray(image_array).save(out_path)
    return out_path


def convert_tif_folder(tif_dir, png_dir):
    '''Convert all GeoTIFF files in a folder to PNG files.

            Parameters:
                tif_dir: Directory containing GeoTIFF tiles.
                png_dir: Directory where PNG images should be saved.

            Returns:
                Number of converted files.
    '''
    png_dir.mkdir(parents=True, exist_ok=True)
    tif_files = sorted(tif_dir.glob("*.tif"))

    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {tif_dir}")

    for tif_path in tif_files:
        out_path = convert_one_tif(tif_path, png_dir)
        print(f"{tif_path.name} -> {out_path}")

    return len(tif_files)


def main():
    parser = ArgumentParser(description="Convert GeoTIFF image tiles to same-name PNG images.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    count = convert_tif_folder(args.input, args.output)
    print(f"converted: {count}")


if __name__ == "__main__":
    main()
