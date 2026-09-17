"""Copy source images to YOLO splits by matching label filenames."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-images", type=Path, required=True)
    return parser.parse_args()


def index_images(source: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.stem, path)
    return index


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    source_images = args.source_images.expanduser().resolve()
    if not source_images.is_dir():
        raise NotADirectoryError(f"Source image directory not found: {source_images}")

    image_index = index_images(source_images)
    copied = 0
    missing: list[tuple[str, str]] = []

    for split in ("train", "val", "test"):
        label_dir = dataset_root / "labels" / split
        image_dir = dataset_root / "images" / split
        if not label_dir.is_dir():
            raise NotADirectoryError(f"Label directory not found: {label_dir}")
        image_dir.mkdir(parents=True, exist_ok=True)

        for label_path in label_dir.glob("*.txt"):
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
            "".join(f"{split}\t{stem}\n" for split, stem in missing),
            encoding="utf-8",
        )
        print(f"Missing-image report: {report}")


if __name__ == "__main__":
    main()

