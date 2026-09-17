"""Minimal IoU and Non-Maximum Suppression implementation for learning."""

from __future__ import annotations

from collections.abc import Sequence


Box = Sequence[float]


def compute_iou(box1: Box, box2: Box) -> float:
    """Return IoU for two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def simple_nms(
    boxes: Sequence[Box],
    scores: Sequence[float],
    iou_threshold: float = 0.5,
) -> list[int]:
    """Return indices kept by class-agnostic greedy NMS."""
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have the same length")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")

    remaining = sorted(range(len(boxes)), key=scores.__getitem__, reverse=True)
    keep: list[int] = []
    while remaining:
        best = remaining.pop(0)
        keep.append(best)
        remaining = [
            index
            for index in remaining
            if compute_iou(boxes[best], boxes[index]) <= iou_threshold
        ]
    return keep


if __name__ == "__main__":
    demo_boxes = [[10, 10, 100, 100], [12, 12, 98, 98], [200, 200, 300, 300]]
    demo_scores = [0.9, 0.8, 0.85]
    print("kept indices:", simple_nms(demo_boxes, demo_scores))

