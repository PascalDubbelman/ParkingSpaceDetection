#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path


DEFAULT_DATA = Path("GeoreferenceTest/202606121032215360267/dataset/data.yaml")


def main():
    parser = ArgumentParser(description="Train a YOLO11 OBB model on the prepared parking-space dataset.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo11s-obb.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Use cpu, 0, 0,1, etc. Omit for Ultralytics auto-selection.")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default=None, help="Training output project directory, e.g. runs/obb")
    parser.add_argument("--name", default=None, help="Training run name, e.g. train")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse the project/name directory instead of creating train2, train3, etc.")
    parser.add_argument("--resume", type=Path, default=None, help="Path to last.pt checkpoint to resume training.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit("ultralytics is not installed in this Python environment") from exc

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


if __name__ == "__main__":
    main()
