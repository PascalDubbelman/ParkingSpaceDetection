#!/usr/bin/env python3
import argparse
import csv
import math
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def find_image_by_name(image_dir, image_name):
    direct = image_dir / image_name
    if direct.exists():
        return direct
    stem = Path(image_name).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_manifest(path, limit=None):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def confidence_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def row_points_px(row):
    return [
        (float(row["x1_px"]), float(row["y1_px"])),
        (float(row["x2_px"]), float(row["y2_px"])),
        (float(row["x3_px"]), float(row["y3_px"])),
        (float(row["x4_px"]), float(row["y4_px"])),
    ]


def distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def unit_vector(a, b):
    length = distance(a, b)
    if length == 0:
        return None
    return ((b[0] - a[0]) / length, (b[1] - a[1]) / length)


def crop_geometry_from_manifest(row):
    points = row_points_px(row)
    p0, p1, p2, p3 = points
    u = unit_vector(p0, p1)
    v = unit_vector(p0, p3)
    if u is None or v is None:
        return None

    pad_x = float(row["padding_x_px"])
    pad_y = float(row["padding_y_px"])
    output_width = int(float(row["rectified_crop_width"]))
    output_height = int(float(row["rectified_crop_height"]))

    q0 = (p0[0] - pad_x * u[0] - pad_y * v[0], p0[1] - pad_x * u[1] - pad_y * v[1])
    q1 = (p1[0] + pad_x * u[0] - pad_y * v[0], p1[1] + pad_x * u[1] - pad_y * v[1])
    q2 = (p2[0] + pad_x * u[0] + pad_y * v[0], p2[1] + pad_x * u[1] + pad_y * v[1])
    q3 = (p3[0] - pad_x * u[0] + pad_y * v[0], p3[1] - pad_x * u[1] + pad_y * v[1])

    return {
        "source_quad": [q0, q1, q2, q3],
        "output_size": (output_width, output_height),
        "rotated_to_portrait": parse_bool(row.get("rotated_to_portrait", "false")),
        "canvas_size": (
            int(float(row.get("canvas_width", 320))),
            int(float(row.get("canvas_height", 480))),
        ),
    }


def rectified_crop(image, geometry):
    from PIL import Image

    q0, q1, q2, q3 = geometry["source_quad"]
    quad_for_pil = [q0[0], q0[1], q3[0], q3[1], q2[0], q2[1], q1[0], q1[1]]
    return image.transform(
        geometry["output_size"],
        Image.Transform.QUAD,
        quad_for_pil,
        resample=Image.Resampling.BICUBIC,
    )


def normalize_to_canvas(image, geometry, background=(244, 241, 234)):
    from PIL import Image, ImageOps

    if geometry["rotated_to_portrait"]:
        image = image.rotate(90, expand=True)
    canvas_size = geometry["canvas_size"]
    fitted = ImageOps.contain(image, canvas_size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", canvas_size, background)
    x = (canvas_size[0] - fitted.width) // 2
    y = (canvas_size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def last_conv_like_modules(torch_model, count):
    import torch.nn as nn

    modules = []
    for name, module in torch_model.named_modules():
        if isinstance(module, nn.Conv2d):
            modules.append((name, module))
    if len(modules) < count:
        raise RuntimeError(f"Found only {len(modules)} Conv2d layers, cannot hook last {count}.")
    return modules[-count:]


def extract_feature_maps(yolo_model, image_path, imgsz, device, layer_count):
    activations = []
    hooks = []
    layers = last_conv_like_modules(yolo_model.model, layer_count)

    def make_hook():
        def hook(_module, _inputs, output):
            if isinstance(output, (list, tuple)):
                output = output[0]
            if hasattr(output, "detach"):
                activations.append(output.detach().float().cpu())
        return hook

    for _name, layer in layers:
        hooks.append(layer.register_forward_hook(make_hook()))

    predict_args = {
        "source": str(image_path),
        "imgsz": imgsz,
        "save": False,
        "verbose": False,
    }
    if device is not None:
        predict_args["device"] = device

    try:
        yolo_model.predict(**predict_args)
    finally:
        for hook in hooks:
            hook.remove()

    if len(activations) < layer_count:
        raise RuntimeError(f"Expected {layer_count} activations, got {len(activations)} for {image_path}")
    return activations[-layer_count:]


def obb_mask(size, points_px, image_size):
    import numpy as np
    from PIL import Image, ImageDraw

    width, height = size
    image_width, image_height = image_size
    scaled_points = [
        (x * width / image_width, y * height / image_height)
        for x, y in points_px
    ]
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(scaled_points, fill=255)
    return np.asarray(mask).astype("float32") / 255.0


def eigencam_for_detection(feature_tensor, row, image_size):
    import numpy as np

    feature = feature_tensor[0].numpy()
    channels, height, width = feature.shape
    mask = obb_mask((width, height), row_points_px(row), image_size)
    roi = mask.reshape(height * width) > 0

    if roi.sum() < 2:
        return np.zeros((height, width), dtype="float32")

    flat = feature.reshape(channels, height * width).T
    roi_flat = flat[roi]
    roi_mean = roi_flat.mean(axis=0, keepdims=True)
    roi_centered = roi_flat - roi_mean

    try:
        _, _, vh = np.linalg.svd(roi_centered, full_matrices=False)
        weights = vh[0]
        cam = (flat - roi_mean) @ weights
    except np.linalg.LinAlgError:
        cam = roi_centered.mean(axis=1)
        full_cam = np.zeros(height * width, dtype="float32")
        full_cam[roi] = cam
        cam = full_cam

    cam = cam.reshape(height, width)
    cam = np.maximum(cam, 0) * mask
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)
    return cam


def resize_cam_to_image(cam, size):
    import numpy as np
    from PIL import Image

    cam_img = Image.fromarray(np.uint8(np.clip(cam, 0, 1) * 255), mode="L")
    cam_img = cam_img.resize(size, resample=Image.Resampling.BICUBIC)
    return np.asarray(cam_img).astype("float32") / 255.0


def feature_maps_for_tile(yolo_model, image_path, imgsz, device, layer_count):
    from PIL import Image

    with Image.open(image_path) as img:
        image_size = img.size

    features = extract_feature_maps(yolo_model, image_path, imgsz, device, layer_count)
    return features, image_size


def detection_cam(features, row, image_size):
    import numpy as np

    cams = [resize_cam_to_image(eigencam_for_detection(feature, row, image_size), image_size) for feature in features]
    cam = np.mean(cams, axis=0)
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam


def heatmap_rgb(cam):
    import numpy as np

    cam = np.clip(cam, 0, 1)
    red = np.clip(1.7 * cam, 0, 1)
    green = np.clip(1.7 * (1 - np.abs(cam - 0.55) / 0.55), 0, 1)
    blue = np.clip(1.7 * (1 - cam), 0, 1) * 0.45
    return np.uint8(np.stack([red, green, blue], axis=-1) * 255)


def overlay_heatmap(image, cam_image, alpha):
    import numpy as np
    from PIL import Image

    base = np.asarray(image.convert("RGB")).astype("float32")
    cam = np.asarray(cam_image.convert("L")).astype("float32") / 255.0
    heat = heatmap_rgb(cam).astype("float32")
    out = base * (1 - alpha) + heat * alpha
    return Image.fromarray(np.uint8(np.clip(out, 0, 255)), mode="RGB")


def prepare_output_dirs(output_dir):
    all_dir = output_dir / "all"
    band_dirs = {
        "high_confidence": output_dir / "high_confidence",
        "medium_confidence": output_dir / "medium_confidence",
        "low_confidence": output_dir / "low_confidence",
        "unknown_confidence": output_dir / "unknown_confidence",
    }
    for folder in [all_dir, *band_dirs.values()]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    return all_dir, band_dirs


def prepare_tile_heatmap_dirs(output_dir, enabled):
    if not enabled:
        return None, None

    tile_dir = output_dir / "tile_heatmaps"
    raw_dir = output_dir / "tile_heatmaps_raw"
    for folder in [tile_dir, raw_dir]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    return tile_dir, raw_dir


def cam_name_from_crop(row):
    crop_name = Path(row["crop_file"]).name
    return f"{Path(crop_name).stem}_cam.png"


def tile_cam_name_from_crop(row, suffix):
    crop_name = Path(row["crop_file"]).name
    return f"{Path(crop_name).stem}{suffix}.png"


def main():
    parser = argparse.ArgumentParser(description="Generate per-detection ROI-targeted Eigen-CAM overlays for YOLO-OBB review crops.")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained YOLO OBB best.pt")
    parser.add_argument("--images", type=Path, required=True, help="Directory containing prediction source images")
    parser.add_argument("--manifest", type=Path, required=True, help="review_crops/manifest.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output directory, e.g. review_crops_cam")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick testing")
    parser.add_argument(
        "--save-tile-heatmaps",
        action="store_true",
        help="Also save full-tile per-detection CAM overlays and raw grayscale heatmaps.",
    )
    args = parser.parse_args()

    try:
        from PIL import Image
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit("This script needs pillow and ultralytics installed.") from exc

    rows = read_manifest(args.manifest, args.limit)
    all_dir, band_dirs = prepare_output_dirs(args.output)
    tile_heatmap_dir, tile_heatmap_raw_dir = prepare_tile_heatmap_dirs(args.output, args.save_tile_heatmaps)

    yolo_model = YOLO(args.model)
    feature_cache = {}
    written_rows = []

    for row in rows:
        image_name = Path(row["source_image"]).name
        image_path = find_image_by_name(args.images, image_name)
        if image_path is None:
            print(f"skip: no source image for {image_name}")
            continue

        geometry = crop_geometry_from_manifest(row)
        if geometry is None:
            print(f"skip: invalid geometry for rank {row.get('rank')}")
            continue

        if image_path not in feature_cache:
            print(f"extracting feature maps: {image_path.name}")
            feature_cache[image_path] = feature_maps_for_tile(yolo_model, image_path, args.imgsz, args.device, args.layers)

        features, image_size = feature_cache[image_path]
        cam = detection_cam(features, row, image_size)
        cam_image = Image.fromarray((cam * 255).astype("uint8"), mode="L")
        tile_cam_path = ""
        tile_cam_raw_path = ""

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if args.save_tile_heatmaps:
                tile_cam_name = tile_cam_name_from_crop(row, "_tile_cam")
                tile_cam_path = tile_heatmap_dir / tile_cam_name
                tile_overlay = overlay_heatmap(image, cam_image, args.alpha)
                tile_overlay.save(tile_cam_path)

                tile_cam_raw_name = tile_cam_name_from_crop(row, "_tile_cam_raw")
                tile_cam_raw_path = tile_heatmap_raw_dir / tile_cam_raw_name
                cam_image.save(tile_cam_raw_path)

            crop = rectified_crop(image, geometry)
            cam_crop = rectified_crop(cam_image, geometry)
            overlay = overlay_heatmap(crop, cam_crop, args.alpha)
            overlay = normalize_to_canvas(overlay, geometry)

        cam_name = cam_name_from_crop(row)
        cam_path = all_dir / cam_name
        overlay.save(cam_path)

        band = row.get("confidence_band", "unknown_confidence")
        shutil.copy2(cam_path, band_dirs.get(band, band_dirs["unknown_confidence"]) / cam_name)

        out_row = dict(row)
        out_row["cam_file"] = str(cam_path)
        out_row["tile_cam_file"] = str(tile_cam_path) if tile_cam_path else ""
        out_row["tile_cam_raw_file"] = str(tile_cam_raw_path) if tile_cam_raw_path else ""
        out_row["cam_alpha"] = args.alpha
        out_row["cam_layers"] = args.layers
        out_row["cam_method"] = "roi_targeted_eigencam"
        written_rows.append(out_row)

    manifest_cam = args.output / "manifest_cam.csv"
    if written_rows:
        fieldnames = list(written_rows[0].keys())
        with manifest_cam.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(written_rows)

    print(f"Wrote {len(written_rows)} CAM overlays to {args.output}")


if __name__ == "__main__":
    main()
