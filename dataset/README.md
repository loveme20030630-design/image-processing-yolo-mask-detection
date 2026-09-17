# Dataset format

原始資料共 4,193 張影像，包含 `with_mask` 與 `without_mask` 兩類。Repository 不附帶原始影像、YOLO 標註與模型權重，避免上傳大型資料並保留資料授權界線。

## Expected layout

```text
yolo_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

每張影像應有一個同檔名的 `.txt` 標註檔。標註格式為：

```text
class_id x_center y_center width height
```

座標與寬高皆需正規化到 0–1；類別編號如下：

| class_id | label |
|---:|---|
| 0 | `without_mask` |
| 1 | `with_mask` |

若標註檔已切分完成、影像仍集中在另一個資料夾，可執行：

```bash
python src/prepare_dataset.py --dataset-root path/to/yolo_dataset --source-images path/to/source_images
```

