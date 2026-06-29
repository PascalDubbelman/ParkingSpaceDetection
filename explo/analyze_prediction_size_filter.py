#!/usr/bin/env python3
"""Quick size/aspect-ratio diagnostics for georeferenced YOLO prediction GeoJSONs.

This script is intended as an exploratory post-processing check. It does not
change the reported model metrics unless the filtered output is explicitly used
in a new validated workflow.

Example:
    python explo/analyze_prediction_size_filter.py \
        explo/image/predictions_nude_OBB.geojson \
        --min-area 5 --max-area 30 \
        --out /tmp/Nude11sOBB_size_filtered.geojson
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def polygon_outer_ring(geometry: dict[str, Any]) -> list[list[float]] | None:
    '''Return the first outer ring from a GeoJSON polygon geometry.

            Parameters:
                geometry: GeoJSON geometry dictionary from a prediction feature.

            Returns:
                List of coordinate pairs from the outer polygon ring, or None.
    '''
    if not geometry:
        return None
    if geometry.get("type") == "Polygon":
        return geometry["coordinates"][0]
    if geometry.get("type") == "MultiPolygon":
        return geometry["coordinates"][0][0]
    return None


def ring_area_m2(ring: list[list[float]]) -> float:
    '''Calculate polygon area using the shoelace formula.

            Parameters:
                ring: List of projected coordinate pairs forming a polygon ring.

            Returns:
                Polygon area in square metres.
    '''
    area = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def distance(a: list[float], b: list[float]) -> float:
    '''Calculate the Euclidean distance between two coordinate pairs.

            Parameters:
                a: First coordinate pair.
                b: Second coordinate pair.

            Returns:
                Distance between the two points in map units.
    '''
    return math.hypot(a[0] - b[0], a[1] - b[1])


def geometry_metrics(geometry: dict[str, Any]) -> dict[str, float] | None:
    '''Calculate size and shape metrics for one prediction geometry.

            Parameters:
                geometry: GeoJSON geometry dictionary from a prediction feature.

            Returns:
                Dictionary with area, long side, short side and aspect ratio,
                or None if the geometry cannot be measured.
    '''
    ring = polygon_outer_ring(geometry)
    if not ring:
        return None

    points = ring[:-1] if ring[0] == ring[-1] else ring
    if len(points) < 4:
        return None

    area = ring_area_m2(ring)
    edges = [distance(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
    side_a = edges[0]
    side_b = edges[1]
    long_side = max(side_a, side_b)
    short_side = min(side_a, side_b)
    aspect = long_side / short_side if short_side else float("inf")

    return {
        "area_m2": area,
        "long_side_m": long_side,
        "short_side_m": short_side,
        "aspect_ratio": aspect,
    }


def percentile(sorted_values: list[float], q: float) -> float:
    '''Return a percentile value from a sorted list.

            Parameters:
                sorted_values: Numeric values sorted from low to high.
                q: Percentile as a fraction between 0 and 1.

            Returns:
                Percentile value, or NaN if the input list is empty.
    '''
    if not sorted_values:
        return float("nan")
    index = int((len(sorted_values) - 1) * q)
    return sorted_values[index]


def passes_filter(
    metrics: dict[str, float],
    min_area: float | None,
    max_area: float | None,
    min_aspect: float | None,
    max_aspect: float | None,
) -> bool:
    '''Check whether prediction metrics pass the selected filters.

            Parameters:
                metrics: Dictionary of calculated geometry metrics.
                min_area: Minimum allowed area in square metres, or None.
                max_area: Maximum allowed area in square metres, or None.
                min_aspect: Minimum allowed long/short side ratio, or None.
                max_aspect: Maximum allowed long/short side ratio, or None.

            Returns:
                True if the detection passes all selected filters, otherwise False.
    '''
    area = metrics["area_m2"]
    aspect = metrics["aspect_ratio"]
    if min_area is not None and area < min_area:
        return False
    if max_area is not None and area > max_area:
        return False
    if min_aspect is not None and aspect < min_aspect:
        return False
    if max_aspect is not None and aspect > max_aspect:
        return False
    return True


def main() -> None:
    '''Run exploratory size and aspect-ratio diagnostics from the command line.

            Parameters:
                None. Command-line arguments are parsed with argparse.

            Returns:
                None. Prints summary statistics and optionally writes a filtered GeoJSON.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("geojson", type=Path)
    parser.add_argument("--min-area", type=float, default=None, help="Minimum detection area in m².")
    parser.add_argument("--max-area", type=float, default=None, help="Maximum detection area in m².")
    parser.add_argument("--min-aspect", type=float, default=None, help="Minimum long/short side ratio.")
    parser.add_argument("--max-aspect", type=float, default=None, help="Maximum long/short side ratio.")
    parser.add_argument("--out", type=Path, default=None, help="Optional filtered GeoJSON output path.")
    args = parser.parse_args()

    data = json.loads(args.geojson.read_text())
    features = data.get("features", [])

    rows: list[tuple[dict[str, Any], dict[str, float]]] = []
    for feature in features:
        metrics = geometry_metrics(feature.get("geometry", {}))
        if metrics:
            rows.append((feature, metrics))

    print(f"Input: {args.geojson}")
    print(f"Features with polygon metrics: {len(rows)} / {len(features)}")

    for key in ["area_m2", "long_side_m", "short_side_m", "aspect_ratio"]:
        values = sorted(metrics[key] for _, metrics in rows)
        print(
            f"{key}: "
            f"min={values[0]:.2f}, "
            f"p05={percentile(values, 0.05):.2f}, "
            f"p25={percentile(values, 0.25):.2f}, "
            f"median={percentile(values, 0.50):.2f}, "
            f"p75={percentile(values, 0.75):.2f}, "
            f"p95={percentile(values, 0.95):.2f}, "
            f"max={values[-1]:.2f}"
        )

    kept = [
        feature
        for feature, metrics in rows
        if passes_filter(metrics, args.min_area, args.max_area, args.min_aspect, args.max_aspect)
    ]
    print(f"Kept after selected filter: {len(kept)}")
    print(f"Removed after selected filter: {len(rows) - len(kept)}")

    if args.out:
        filtered = dict(data)
        filtered["features"] = kept
        args.out.write_text(json.dumps(filtered, indent=2))
        print(f"Wrote filtered GeoJSON: {args.out}")


if __name__ == "__main__":
    main()
