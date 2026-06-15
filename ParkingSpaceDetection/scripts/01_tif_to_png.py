#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import rasterio
from PIL import Image


DEFAULT_INPUT = Path("GeoreferenceTest/202606121032215360267/images")
DEFAULT_OUTPUT = Path("GeoreferenceTest/202606121032215360267/png")


def convert_tif(tif_path, out_dir):
    out_path = out_dir / f"{tif_path.stem}.png"
    with rasterio.open(tif_path) as src:
        if src.count < 3:
            raise ValueError(f"{tif_path.name} has {src.count} band(s), expected at least 3")
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
    Image.fromarray(rgb).save(out_path)
    return out_path


def main():
    parser = ArgumentParser(description="Convert GeoTIFF image tiles to same-name PNGs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    tif_files = sorted(args.input.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {args.input}")

    for tif_path in tif_files:
        out_path = convert_tif(tif_path, args.output)
        print(f"{tif_path.name} -> {out_path}")

    print(f"converted: {len(tif_files)}")


if __name__ == "__main__":
    main()
