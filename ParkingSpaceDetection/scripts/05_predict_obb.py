#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path


DEFAULT_IMAGES = Path("GeoreferenceTest/202606121032215360267/dataset/images/val")


def main():
    parser = ArgumentParser(description="Run YOLO OBB prediction and save txt outputs.")
    parser.add_argument("--model", required=True, help="Path to trained best.pt")
    parser.add_argument("--source", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="Use cpu, 0, 0,1, etc. Omit for Ultralytics auto-selection.")
    parser.add_argument("--project", default="runs/obb")
    parser.add_argument("--name", default="predict")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit("ultralytics is not installed in this Python environment") from exc

    model = YOLO(args.model)
    predict_args = {
        "source": str(args.source),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "save": True,
        "save_txt": True,
        "save_conf": True,
        "project": args.project,
        "name": args.name,
    }
    if args.device is not None:
        predict_args["device"] = args.device
    model.predict(**predict_args)


if __name__ == "__main__":
    main()
