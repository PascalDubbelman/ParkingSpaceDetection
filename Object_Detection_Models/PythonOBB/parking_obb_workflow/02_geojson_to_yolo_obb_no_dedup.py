#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import json

import rasterio
from shapely.geometry import Polygon, shape


def get_tile_polygon(dataset):
    '''Create a map polygon for the full GeoTIFF tile.

            Parameters:
                dataset: Open Rasterio GeoTIFF dataset.

            Returns:
                Shapely polygon of the tile footprint in map coordinates.
    '''
    width = dataset.width
    height = dataset.height
    pixel_corners = [(0, 0), (width, 0), (width, height), (0, height)]
    map_corners = [dataset.transform * point for point in pixel_corners]
    return Polygon(map_corners)


def get_rectangle_corners(geometry):
    '''Get four corner points from a parking polygon.

            Parameters:
                geometry: Shapely geometry from the GeoJSON.

            Returns:
                List of four map-coordinate corners, or None.
    '''
    if geometry.geom_type == "MultiPolygon":
        geometry = max(geometry.geoms, key=lambda item: item.area)

    if geometry.geom_type != "Polygon":
        return None

    corners = list(geometry.exterior.coords)[:-1]

    if len(corners) != 4:
        rectangle = geometry.minimum_rotated_rectangle
        corners = list(rectangle.exterior.coords)[:-1]

    if len(corners) != 4:
        return None

    return corners


def convert_corners_to_yolo_row(corners, dataset, class_id):
    '''Convert four map-coordinate corners to one YOLO OBB label row.

            Parameters:
                corners: Four corner points in map coordinates.
                dataset: Open Rasterio GeoTIFF dataset.
                class_id: YOLO class id.

            Returns:
                YOLO OBB label row, or None if outside the tile.
    '''
    inverse_transform = ~dataset.transform
    values = []

    for map_x, map_y in corners:
        pixel_x, pixel_y = inverse_transform * (map_x, map_y)
        norm_x = pixel_x / dataset.width
        norm_y = pixel_y / dataset.height

        if norm_x < 0 or norm_x > 1 or norm_y < 0 or norm_y > 1:
            return None

        values.extend([norm_x, norm_y])

    return f"{class_id} " + " ".join(f"{value:.6f}" for value in values)


def load_parking_polygons(geojson_path):
    '''Read parking polygons from a GeoJSON file.

            Parameters:
                geojson_path: Path to the annotation GeoJSON.

            Returns:
                List of Shapely geometries.
    '''
    data = json.loads(geojson_path.read_text())
    polygons = []

    for feature in data.get("features", []):
        polygons.append(shape(feature["geometry"]))

    return polygons


def write_labels_for_tile(tif_path, parking_polygons, output_dir, class_id, min_visible):
    '''Write one YOLO OBB TXT file for one GeoTIFF tile.

            Parameters:
                tif_path: Path to the GeoTIFF tile.
                parking_polygons: List of Shapely parking polygons.
                output_dir: Directory where YOLO TXT labels are saved.
                class_id: YOLO class id.
                min_visible: Minimum visible fraction required inside the tile.

            Returns:
                Number of labels written for this tile.
    '''
    label_rows = []

    with rasterio.open(tif_path) as dataset:
        tile_footprint = get_tile_polygon(dataset)

        for polygon in parking_polygons:
            if polygon.is_empty or not polygon.intersects(tile_footprint):
                continue

            visible_area = polygon.intersection(tile_footprint).area
            visible_ratio = visible_area / polygon.area

            if visible_ratio < min_visible:
                continue

            corners = get_rectangle_corners(polygon)
            if corners is None:
                continue

            row = convert_corners_to_yolo_row(corners, dataset, class_id)
            if row is not None:
                label_rows.append(row)

    out_path = output_dir / f"{tif_path.stem}.txt"
    out_path.write_text("\n".join(label_rows) + ("\n" if label_rows else ""))

    print(f"{tif_path.name} -> {out_path} ({len(label_rows)} labels)")
    return len(label_rows)


def main():
    parser = ArgumentParser(description="Convert GeoJSON parking polygons to YOLO OBB labels without deduplication.")
    parser.add_argument("--geojson", type=Path, required=True)
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--min-visible", type=float, default=0.90)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    parking_polygons = load_parking_polygons(args.geojson)
    tif_files = sorted(args.tiles.glob("*.tif"))

    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {args.tiles}")

    total_labels = 0
    for tif_path in tif_files:
        total_labels += write_labels_for_tile(
            tif_path,
            parking_polygons,
            args.output,
            args.class_id,
            args.min_visible,
        )

    print(f"tiles: {len(tif_files)}")
    print(f"labels: {total_labels}")


if __name__ == "__main__":
    main()
