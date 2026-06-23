#!/usr/bin/env python3
''' Notebook 4: Combine Geojson layers

This script is used for round 2 and the testing tiles.
This script takes all the separately created Geojson layers in the annotation round 2 tutorial and creates a single Geojson out of it.
The script was written for a project commissioned by the Municipality of Wageningen in relation to the course Remote Sensing and GIS integration (GRS60312).
The authors of the script are Polly Cheung, Pascal Dubbelman, Anthony Jansen, Iris Lagemaat, and Susanna van de Wetering.

Last edited: 23/06/2026
'''
# Install packages 
from argparse import ArgumentParser
from pathlib import Path
import json


# CHANGE HERE if you want to run the script without command-line arguments.
# Put all GeoJSON annotation files in INPUT_FOLDER.
# The merged result will be written to OUTPUT_GEOJSON.
INPUT_FOLDER = Path("GeoreferenceTest/202606121032215360267/third_round/all_geojsons")
OUTPUT_GEOJSON = Path("GeoreferenceTest/202606121032215360267/third_round/combined.geojson")


def get_sort_key(geojson_path):
    '''Create a numerical sort key from a GeoJSON filename.

            Parameters:
                geojson_path: Path to one GeoJSON file.

            Returns:
                Tuple of numbers found in the filename.
    '''
    filename = geojson_path.stem
    filename = filename.replace("image_", "")
    parts = filename.split("_")

    numbers = []
    for part in parts:
        if part.isdigit():
            numbers.append(int(part))

    return tuple(numbers)


def find_geojson_files(input_folder):
    '''Find all GeoJSON files in one folder.

            Parameters:
                input_folder: Folder containing GeoJSON files.

            Returns:
                Sorted list of GeoJSON paths.
    '''
    geojson_files = []
    geojson_files.extend(input_folder.glob("*.geojson"))
    geojson_files.extend(input_folder.glob("*.json"))
    geojson_files = sorted(geojson_files, key=get_sort_key)

    if not geojson_files:
        raise FileNotFoundError(f"No .geojson or .json files found in {input_folder}")

    return geojson_files


def read_geojson(geojson_path):
    '''Read one GeoJSON file.

            Parameters:
                geojson_path: Path to the GeoJSON file.

            Returns:
                GeoJSON data as a Python dictionary.
    '''
    return json.loads(geojson_path.read_text())


def add_source_name(feature, geojson_path):
    '''Add the source GeoJSON filename to one feature.

            Parameters:
                feature: One GeoJSON feature.
                geojson_path: Path to the file this feature came from.

            Returns:
                GeoJSON feature with source_geojson in properties.
    '''
    feature = dict(feature)
    properties = dict(feature.get("properties") or {})
    properties["source_geojson"] = geojson_path.name
    feature["properties"] = properties
    return feature


def collect_features(geojson_files):
    '''Collect all features from all GeoJSON files.

            Parameters:
                geojson_files: List of GeoJSON file paths.

            Returns:
                Features list and first CRS found in the input files.
    '''
    all_features = []
    first_crs = None

    for geojson_path in geojson_files:
        data = read_geojson(geojson_path)

        if first_crs is None:
            first_crs = data.get("crs")

        for feature in data.get("features", []):
            feature = add_source_name(feature, geojson_path)
            all_features.append(feature)

    return all_features, first_crs


def write_combined_geojson(features, crs, output_geojson):
    '''Write all features into one combined GeoJSON file.

            Parameters:
                features: List of GeoJSON features.
                crs: CRS copied from the first input file, if available.
                output_geojson: Path where the combined GeoJSON is saved.

            Returns:
                None.
    '''
    combined = {
        "type": "FeatureCollection",
        "name": output_geojson.stem,
        "features": features,
    }

    if crs is not None:
        combined["crs"] = crs

    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    output_geojson.write_text(json.dumps(combined, indent=2) + "\n")


def main():
    parser = ArgumentParser(description="Combine all GeoJSON files in one folder into one GeoJSON layer.")
    parser.add_argument("--input", type=Path, default=INPUT_FOLDER)
    parser.add_argument("--output", type=Path, default=OUTPUT_GEOJSON)
    args = parser.parse_args()

    geojson_files = find_geojson_files(args.input)
    features, crs = collect_features(geojson_files)
    write_combined_geojson(features, crs, args.output)

    print(f"files: {len(geojson_files)}")
    print(f"features: {len(features)}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
