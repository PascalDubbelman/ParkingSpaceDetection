#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path


def train_obb_model(args):
    '''Train a YOLO OBB model.

            Parameters:
                args: Command-line arguments for model, data, and training settings.

            Returns:
                None.
    '''
    from ultralytics import YOLO

    if args.resume is not None:
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    model = YOLO(args.model)

    train_args = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "single_cls": True,
        "rect": True,
        "patience": args.patience,
        "exist_ok": args.exist_ok,
    }

    if args.device is not None:
        train_args["device"] = args.device
    if args.project is not None:
        train_args["project"] = args.project
    if args.name is not None:
        train_args["name"] = args.name

    model.train(**train_args)


def main():
    parser = ArgumentParser(description="Train YOLO OBB on the parking-space dataset.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo11s-obb.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    try:
        train_obb_model(args)
    except ModuleNotFoundError as exc:
        raise SystemExit("ultralytics is not installed in this Python environment") from exc


if __name__ == "__main__":
    main()
