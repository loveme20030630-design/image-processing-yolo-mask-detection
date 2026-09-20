# Dataset format

本專案的原始影像取自 Kaggle「Face Mask Detection ~12K Images Dataset」。原資料主要以 `WithMask` 與 `WithoutMask` 類別整理；本專案為進行 YOLO 物件偵測，另使用 LabelImg 手動建立人臉區域與口罩狀態的 bounding box 標註。

本次實驗實際使用 2,002 張具有 YOLO 格式標註的影像：

| Split | Images | Label files |
|---|---:|---:|
| train | 1,401 | 1,401 |
| val | 300 | 300 |
| test | 301 | 301 |
| **Total** | **2,002** | **2,002** |

完整原始影像、2,002 份完整標註檔與模型權重未放入公開 repository。

---

## Important: labels must already exist

`src/prepare_dataset.py` 不會建立、推測或自動產生 bounding-box annotation。

執行資料整理程式前，必須先準備完成 YOLO 格式的 `.txt` 標註檔，並放入：

```text
labels/train/
labels/val/
labels/test/
```

每個標註檔必須與對應影像使用完全相同的檔名主體。

例如：

```text
1234.png    <-> 1234.txt
5678_1.jpg  <-> 5678_1.txt
```

若檔名主體不同，`prepare_dataset.py` 將無法找到對應影像。

---

## Expected layout

```text
yolo_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── labels/
    ├── train/
    ├── val/
    └── test/
```

---

## YOLO label format

每張影像應有一個對應的 `.txt` 標註檔。

每一行代表一個 bounding box：

```text
class_id x_center y_center width height
```

其中：

- `class_id`：目標類別編號
- `x_center`：bounding box 中心點 X 座標
- `y_center`：bounding box 中心點 Y 座標
- `width`：bounding box 寬度
- `height`：bounding box 高度

`x_center`、`y_center`、`width` 與 `height` 均需正規化至 0–1。

目前 repository 使用的類別編號為：

| class_id | label |
|---:|---|
| 0 | `without_mask` |
| 1 | `with_mask` |

例如：

```text
0 0.495798 0.752101 0.857143 0.495798
```

---

## Image and label filename matching

資料整理流程以標註檔名作為影像搜尋依據。

例如：

```text
labels/train/1234.txt
```

必須能找到：

```text
1234.png
```

或其他支援的影像格式，例如：

```text
1234.jpg
1234.jpeg
1234.bmp
1234.webp
```

只要副檔名不同沒有關係，但檔名主體必須一致。

例如標註檔為：

```text
5678_1.txt
```

則來源影像也必須使用：

```text
5678_1.png
```

或：

```text
5678_1.jpg
```

---

## Preparing images from existing labels

若 `labels/train`、`labels/val`、`labels/test` 已經完成切分，但影像仍集中存放在另一個來源資料夾，可執行：

```bash
python src/prepare_dataset.py \
  --dataset-root path/to/yolo_dataset \
  --source-images path/to/source_images
```

其中：

`--dataset-root` 應指向：

```text
yolo_dataset/
```

`--source-images` 應指向包含來源影像的資料夾。

---

## What prepare_dataset.py does

`prepare_dataset.py` 的用途是依照既有 label 檔名整理影像，不負責建立 annotation。

程式流程為：

1. 讀取 `labels/train`、`labels/val`、`labels/test` 中的 `.txt` 標註檔。
2. 取得每個標註檔的檔名主體。
3. 在來源影像資料夾中遞迴搜尋同名影像。
4. 將找到的影像複製到對應的 `images/train`、`images/val` 或 `images/test`。
5. 記錄找不到對應影像的標註項目。

例如：

```text
labels/train/1234.txt
```

程式會搜尋：

```text
1234.jpg
1234.jpeg
1234.png
1234.bmp
1234.webp
```

找到後複製至：

```text
images/train/
```

---

## Missing-image report

若某個 label 找不到對應影像，程式會建立：

```text
missing_images.tsv
```

內容會記錄 split 與找不到的檔名主體。

開始訓練前應先確認這些缺失項目。

常見原因包括：

- 影像檔名與 label 檔名不一致
- 影像未放入指定來源資料夾
- label 被保留，但原始影像已刪除
- 檔名中存在額外字元
- split 整理錯誤

---

## data.yaml

完成資料整理後，可複製：

```text
config/data.example.yaml
```

為：

```text
config/data.yaml
```

並修改資料集路徑。

範例：

```yaml
path: /absolute/path/to/yolo_dataset

train: images/train
val: images/val
test: images/test

names:
  0: without_mask
  1: with_mask
```

其中類別順序必須與標註檔中的 `class_id` 保持一致。

---

## Before training

開始 YOLOv8 訓練前，應確認以下事項：

- `images/train` 與 `labels/train` 的資料可一一對應
- `images/val` 與 `labels/val` 的資料可一一對應
- `images/test` 與 `labels/test` 的資料可一一對應
- image 與 label 的檔名主體完全一致
- label 內容符合 YOLO 格式
- bounding-box 座標已正規化至 0–1
- `class_id` 僅使用已定義的類別
- `data.yaml` 的資料集路徑正確
- `data.yaml` 的類別順序與 annotation 一致
- `missing_images.tsv` 中沒有尚未處理的必要資料

---

## Dataset used in this project

本次實驗資料量為：

```text
Train:       1,401
Validation:    300
Test:          301
------------------
Total:        2,002
```

對應 YOLO label 數量亦為：

```text
Train:       1,401
Validation:    300
Test:          301
------------------
Total:        2,002
```

本 repository 不公開完整資料集內容，但保留資料格式、設定範例與資料整理流程，以利理解專案架構與執行方式。
