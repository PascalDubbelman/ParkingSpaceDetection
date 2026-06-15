#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import json

import rasterio


DEFAULT_TILES = Path("GeoreferenceTest/202606121032215360267/images")


def pixel_to_map(transform, px, py):
    return list(transform * (px, py))


def parse_prediction_row(line):
    parts = line.split()
    if len(parts) not in (9, 10):
        raise ValueError(f"expected 9 or 10 columns for YOLO OBB prediction, got {len(parts)}: {line}")
    class_id = int(float(parts[0]))
    coords = [float(v) for v in parts[1:9]]
    confidence = float(parts[9]) if len(parts) == 10 else None
    return class_id, coords, confidence


def feature_from_row(row, txt_path, dataset):
    class_id, coords, confidence = parse_prediction_row(row)
    points = []
    for x_norm, y_norm in zip(coords[0::2], coords[1::2]):
        px = x_norm * dataset.width
        py = y_norm * dataset.height
        points.append(pixel_to_map(dataset.transform, px, py))
    points.append(points[0])

    props = {
        "class": class_id,
        "source": txt_path.stem,
        "tile": f"{txt_path.stem}.tif",
    }
    if confidence is not None:
        props["confidence"] = confidence

    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "Polygon", "coordinates": [points]},
    }


def main():
    parser = ArgumentParser(description="Convert YOLO OBB prediction txt files to one RD New GeoJSON.")
    parser.add_argument("predictions", type=Path, help="Folder with YOLO OBB prediction .txt files")
    parser.add_argument("--tiles", type=Path, default=DEFAULT_TILES)
    parser.add_argument("-o", "--out", type=Path, default=Path("predictions.geojson"))
    args = parser.parse_args()

    features = []
    for txt_path in sorted(args.predictions.glob("*.txt")):
        tif_path = args.tiles / f"{txt_path.stem}.tif"
        if not tif_path.exists():
            raise FileNotFoundError(f"Missing matching GeoTIFF for {txt_path.name}: {tif_path}")
        with rasterio.open(tif_path) as dataset:
            for line in txt_path.read_text().splitlines():
                if line.strip():
                    features.append(feature_from_row(line, txt_path, dataset))

    geojson = {
        "type": "FeatureCollection",
        "name": args.out.stem,
        "crs": {"type": "name", "properties": {"name": "EPSG:28992"}},
        "properties": {"crs": "EPSG:28992 - Amersfoort / RD New"},
        "features": features,
    }
    args.out.write_text(json.dumps(geojson, indent=2) + "\n")
    print(f"{args.out} ({len(features)} features)")


if __name__ == "__main__":
    main()
