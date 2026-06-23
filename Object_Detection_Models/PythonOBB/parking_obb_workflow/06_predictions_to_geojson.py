#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import json

import rasterio


def parse_prediction_line(line):
    '''Read one YOLO OBB prediction row.

            Parameters:
                line: One row from a YOLO prediction TXT file.

            Returns:
                class id, normalized coordinates, and confidence value.
    '''
    parts = line.split()

    if len(parts) not in (9, 10):
        raise ValueError(f"expected 9 or 10 columns, got {len(parts)}: {line}")

    class_id = int(float(parts[0]))
    coordinates = [float(value) for value in parts[1:9]]
    confidence = float(parts[9]) if len(parts) == 10 else None

    return class_id, coordinates, confidence


def normalized_points_to_map_points(coordinates, dataset):
    '''Convert normalized YOLO OBB points to map coordinates.

            Parameters:
                coordinates: Eight normalized YOLO OBB coordinates.
                dataset: Open Rasterio GeoTIFF dataset.

            Returns:
                Closed polygon coordinate list in map coordinates.
    '''
    points = []

    for norm_x, norm_y in zip(coordinates[0::2], coordinates[1::2]):
        pixel_x = norm_x * dataset.width
        pixel_y = norm_y * dataset.height
        map_x, map_y = dataset.transform * (pixel_x, pixel_y)
        points.append([map_x, map_y])

    points.append(points[0])
    return points


def make_geojson_feature(line, txt_path, dataset):
    '''Convert one YOLO prediction row to one GeoJSON feature.

            Parameters:
                line: One prediction row.
                txt_path: Path to the prediction TXT file.
                dataset: Matching GeoTIFF dataset.

            Returns:
                GeoJSON feature dictionary.
    '''
    class_id, coordinates, confidence = parse_prediction_line(line)
    points = normalized_points_to_map_points(coordinates, dataset)

    properties = {
        "class": class_id,
        "source": txt_path.stem,
        "tile": f"{txt_path.stem}.tif",
    }

    if confidence is not None:
        properties["confidence"] = confidence

    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "Polygon", "coordinates": [points]},
    }


def convert_predictions_to_features(prediction_dir, tile_dir):
    '''Convert all prediction TXT files to GeoJSON features.

            Parameters:
                prediction_dir: Folder containing YOLO prediction TXT files.
                tile_dir: Folder containing matching GeoTIFF tiles.

            Returns:
                List of GeoJSON feature dictionaries.
    '''
    features = []

    for txt_path in sorted(prediction_dir.glob("*.txt")):
        tif_path = tile_dir / f"{txt_path.stem}.tif"

        if not tif_path.exists():
            raise FileNotFoundError(f"Missing matching GeoTIFF for {txt_path.name}: {tif_path}")

        with rasterio.open(tif_path) as dataset:
            for line in txt_path.read_text().splitlines():
                if line.strip():
                    features.append(make_geojson_feature(line, txt_path, dataset))

    return features


def write_geojson(features, out_path):
    '''Write prediction features to a GeoJSON file in RD New.

            Parameters:
                features: List of GeoJSON feature dictionaries.
                out_path: Output GeoJSON path.

            Returns:
                None.
    '''
    geojson = {
        "type": "FeatureCollection",
        "name": out_path.stem,
        "crs": {"type": "name", "properties": {"name": "EPSG:28992"}},
        "properties": {"crs": "EPSG:28992 - Amersfoort / RD New"},
        "features": features,
    }

    out_path.write_text(json.dumps(geojson, indent=2) + "\n")


def main():
    parser = ArgumentParser(description="Convert YOLO OBB predictions to RD New GeoJSON.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("-o", "--out", type=Path, default=Path("predictions.geojson"))
    args = parser.parse_args()

    features = convert_predictions_to_features(args.predictions, args.tiles)
    write_geojson(features, args.out)

    print(f"{args.out} ({len(features)} features)")


if __name__ == "__main__":
    main()
