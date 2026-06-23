#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path


def predict_with_obb_model(args):
    '''Run YOLO OBB prediction and save images and TXT labels.

            Parameters:
                args: Command-line arguments for model, source images, and output path.

            Returns:
                None.
    '''
    from ultralytics import YOLO

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


def main():
    parser = ArgumentParser(description="Run YOLO OBB prediction and save TXT outputs.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/obb")
    parser.add_argument("--name", default="predict")
    args = parser.parse_args()

    try:
        predict_with_obb_model(args)
    except ModuleNotFoundError as exc:
        raise SystemExit("ultralytics is not installed in this Python environment") from exc


if __name__ == "__main__":
    main()
