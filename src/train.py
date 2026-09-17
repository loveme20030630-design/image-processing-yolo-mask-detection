"""Train an Ultralytics YOLO model for mask detection."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="Pretrained model or weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs/mask_detection"))
    parser.add_argument("--name", default=None, help="Run name; defaults to the model stem")
    parser.add_argument("--device", default=None, help="Examples: 0, cpu; auto-selected if omitted")
    parser.add_argument("--save-period", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    run_name = args.name or Path(args.model).stem
    args.project.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=run_name,
        device=device,
        save_period=args.save_period,
        workers=args.workers,
        verbose=True,
    )


if __name__ == "__main__":
    main()

