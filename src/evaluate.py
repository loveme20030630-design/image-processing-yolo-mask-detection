"""Evaluate trained YOLO weights on a validation or test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    data = args.data.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data}")

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    metrics = YOLO(str(weights)).val(
        data=str(data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
    )

    print(f"split: {args.split}")
    print(f"precision: {metrics.box.mp:.6f}")
    print(f"recall: {metrics.box.mr:.6f}")
    print(f"mAP@50: {metrics.box.map50:.6f}")
    print(f"mAP@50-95: {metrics.box.map:.6f}")


if __name__ == "__main__":
    main()

