"""Run mask detection on an image, directory, video, or webcam source."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Path, URL, directory, video, or webcam index")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", type=Path, default=Path("runs/predict"))
    parser.add_argument("--name", default="mask_detection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    source: str | int = int(args.source) if args.source.isdigit() else args.source
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")

    YOLO(str(weights)).predict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        save=True,
        project=str(args.project),
        name=args.name,
    )


if __name__ == "__main__":
    main()

