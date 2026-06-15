#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import json

from shapely.geometry import shape


def confidence(feature):
    value = feature.get("properties", {}).get("confidence")
    return float(value) if value is not None else 1.0


def class_id(feature):
    return feature.get("properties", {}).get("class", 0)


def polygon_iou(a, b):
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.union(b).area
    return inter / union if union else 0.0


def dedupe(features, iou_threshold):
    items = [(feature, shape(feature["geometry"])) for feature in features]
    items.sort(key=lambda item: confidence(item[0]), reverse=True)

    kept = []
    for feature, geom in items:
        duplicate = False
        for kept_feature, kept_geom in kept:
            if class_id(feature) != class_id(kept_feature):
                continue
            if polygon_iou(geom, kept_geom) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append((feature, geom))
    return [feature for feature, _ in kept]


def main():
    parser = ArgumentParser(description="Dedupe overlapping RD New prediction polygons with polygon IoU NMS.")
    parser.add_argument("geojson", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=Path("predictions_deduped.geojson"))
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()

    data = json.loads(args.geojson.read_text())
    original = data.get("features", [])
    data["features"] = dedupe(original, args.iou)
    data.setdefault("properties", {})
    data["properties"]["dedupe_iou"] = args.iou
    data["properties"]["features_before_dedupe"] = len(original)
    data["properties"]["features_after_dedupe"] = len(data["features"])

    args.out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{args.out} ({len(original)} -> {len(data['features'])} features)")


if __name__ == "__main__":
    main()
