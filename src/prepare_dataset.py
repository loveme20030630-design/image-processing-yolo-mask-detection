"""Organize source images into YOLO splits by matching existing label filenames.

This script does NOT create bounding-box annotations.

Before running this script, YOLO-format .txt label files must already exist in:

    labels/train/
    labels/val/
    labels/test/

Each label filename stem must match its corresponding image filename stem.

Example:

    1234.txt <-> 1234.png
    5678_1.txt <-> 5678_1.jpg

The script searches for the corresponding source image and copies it into
images/train, images/val, or images/test according to the existing label split.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "YOLO dataset root containing labels/train, labels/val, "
            "and labels/test."
        ),
    )

    parser.add_argument(
        "--source-images",
        type=Path,
        required=True,
        help=(
            "Directory containing source images. "
            "Subdirectories are searched recursively."
        ),
    )

    return parser.parse_args()


def index_images(source: Path) -> dict[str, Path]:
    """Build an image index and reject duplicate filename stems."""

    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}

    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        existing = index.get(path.stem)

        if existing is None:
            index[path.stem] = path
            continue

        duplicates.setdefault(path.stem, [existing]).append(path)

    if duplicates:
        details = "\n".join(
            f"{stem}: " + " | ".join(str(path) for path in paths)
            for stem, paths in sorted(duplicates.items())
        )

        raise ValueError(
            "Duplicate image filename stems found.\n"
            "Each label stem must map to exactly one source image.\n"
            f"{details}"
        )

    return index

def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    source_images = args.source_images.expanduser().resolve()

    if not source_images.is_dir():
        raise NotADirectoryError(
            f"Source image directory not found: {source_images}"
        )

    image_index = index_images(source_images)

    copied = 0
    missing: list[tuple[str, str]] = []

    for split in ("train", "val", "test"):
        label_dir = dataset_root / "labels" / split
        image_dir = dataset_root / "images" / split

        if not label_dir.is_dir():
            raise NotADirectoryError(
                f"Label directory not found: {label_dir}\n"
                "This script does not create annotations. "
                "Prepare YOLO .txt labels before running it."
            )

        label_paths = sorted(label_dir.glob("*.txt"))

        if not label_paths:
            raise FileNotFoundError(
                f"No YOLO .txt label files found in: {label_dir}\n"
                "This script does not generate bounding-box annotations."
            )

        image_dir.mkdir(parents=True, exist_ok=True)

        for label_path in label_paths:
            source = image_index.get(label_path.stem)

            if source is None:
                missing.append((split, label_path.stem))
                continue

            destination = image_dir / source.name

            if not destination.exists():
                shutil.copy2(source, destination)
                copied += 1

    print(f"Copied images: {copied}")
    print(f"Missing images: {len(missing)}")

    if missing:
        report = dataset_root / "missing_images.tsv"

        report.write_text(
            "".join(
                f"{split}\t{stem}\n"
                for split, stem in missing
            ),
            encoding="utf-8",
        )

        print(f"Missing-image report: {report}")
        print(
            "Check that each label filename stem exactly matches "
            "its corresponding image filename stem before training."
        )


if __name__ == "__main__":
    main()
