#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import json

import rasterio
from shapely.geometry import Polygon, shape


DEFAULT_GEOJSON = Path("GeoreferenceTest/image_21_30.geojson")
DEFAULT_TILES = Path("GeoreferenceTest/202606121032215360267/images")
DEFAULT_OUTPUT = Path("GeoreferenceTest/202606121032215360267/obb_labels")


def tile_polygon(dataset):
    width, height = dataset.width, dataset.height
    corners = [(0, 0), (width, 0), (width, height), (0, height)]
    return Polygon([dataset.transform * xy for xy in corners])


def rectangle_corners(geometry):
    geom = geometry
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type != "Polygon":
        return None

    coords = list(geom.exterior.coords)[:-1]
    if len(coords) != 4:
        coords = list(geom.minimum_rotated_rectangle.exterior.coords)[:-1]
    if len(coords) != 4:
        return None
    return coords


def normalized_obb_row(corners, dataset, class_id):
    inv = ~dataset.transform
    values = []
    for x, y in corners:
        px, py = inv * (x, y)
        nx = px / dataset.width
        ny = py / dataset.height
        if nx < 0 or nx > 1 or ny < 0 or ny > 1:
            return None
        values.extend([nx, ny])
    return f"{class_id} " + " ".join(f"{v:.6f}" for v in values)


def load_features(path):
    data = json.loads(path.read_text())
    return [shape(feature["geometry"]) for feature in data.get("features", [])]


def main():
    parser = ArgumentParser(description="Convert RD New GeoJSON parking polygons to YOLO OBB labels per GeoTIFF tile.")
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON)
    parser.add_argument("--tiles", type=Path, default=DEFAULT_TILES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--min-visible", type=float, default=0.70)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    parking_polygons = load_features(args.geojson)
    tif_files = sorted(args.tiles.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {args.tiles}")

    total = 0
    for tif_path in tif_files:
        rows = []
        with rasterio.open(tif_path) as dataset:
            footprint = tile_polygon(dataset)
            for polygon in parking_polygons:
                if polygon.is_empty or not polygon.intersects(footprint):
                    continue
                visible_ratio = polygon.intersection(footprint).area / polygon.area
                if visible_ratio < args.min_visible:
                    continue
                corners = rectangle_corners(polygon)
                if not corners:
                    continue
                row = normalized_obb_row(corners, dataset, args.class_id)
                if row:
                    rows.append(row)

        out_path = args.output / f"{tif_path.stem}.txt"
        out_path.write_text("\n".join(rows) + ("\n" if rows else ""))
        total += len(rows)
        print(f"{tif_path.name} -> {out_path} ({len(rows)} labels)")

    print(f"tiles: {len(tif_files)}")
    print(f"labels: {total}")


if __name__ == "__main__":
    main()
