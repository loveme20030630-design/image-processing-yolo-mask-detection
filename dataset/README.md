# Dataset Format

本專案的原始影像取自 Kaggle **Face Mask Detection ~12K Images Dataset**。

原始資料主要以 `WithMask` 與 `WithoutMask` 類別整理；本專案為進行 YOLO 物件偵測，另外使用 LabelImg 手動建立人臉區域與口罩狀態的 Bounding Box 標註。

本次實驗實際使用 **2,002 張具有 YOLO 格式標註的影像**：

| Split     |    Images | Label files |
| --------- | --------: | ----------: |
| train     |     1,401 |       1,401 |
| val       |       300 |         300 |
| test      |       301 |         301 |
| **Total** | **2,002** |   **2,002** |

完整原始影像、2,002 份完整標註檔與模型權重未放入公開 repository。

---

## Important: Labels Must Already Exist

`src/prepare_dataset.py` 不會建立、推測或自動產生 Bounding Box annotation。

執行資料整理程式前，必須先準備完成 YOLO 格式的 `.txt` 標註檔，並放入：

```text
labels/train/
labels/val/
labels/test/
```

每個標註檔必須與對應影像使用完全相同的 filename stem。

例如：

```text
1234.png    <-> 1234.txt
5678_1.jpg  <-> 5678_1.txt
```

若 filename stem 不同，`prepare_dataset.py` 將無法找到對應影像。

---

## Expected Layout

預期資料結構如下：

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

## YOLO Label Format

每張影像應有一個對應的 `.txt` 標註檔。

每一行代表一個 Bounding Box：

```text
class_id x_center y_center width height
```

其中：

* `class_id`：目標類別編號
* `x_center`：Bounding Box 中心點 X 座標
* `y_center`：Bounding Box 中心點 Y 座標
* `width`：Bounding Box 寬度
* `height`：Bounding Box 高度

`x_center`、`y_center`、`width` 與 `height` 均需正規化至 0–1。

目前 repository 使用的類別編號為：

| class_id | label          |
| -------: | -------------- |
|        0 | `without_mask` |
|        1 | `with_mask`    |

例如：

```text
0 0.495798 0.752101 0.857143 0.495798
```

代表該 Bounding Box 類別為 `without_mask`。

---

## Image and Label Filename Matching

資料整理流程以標註檔的 filename stem 作為影像搜尋依據。

例如：

```text
labels/train/1234.txt
```

必須能找到具有相同 filename stem 的影像，例如：

```text
1234.png
```

或其他支援格式：

```text
1234.jpg
1234.jpeg
1234.bmp
1234.webp
```

標註檔與影像的副檔名可以不同，但 filename stem 必須一致。

例如標註檔為：

```text
5678_1.txt
```

則來源影像可以是：

```text
5678_1.png
```

或：

```text
5678_1.jpg
```

---

## Duplicate Filename Stem

為避免同一個 Label 對應到多張來源影像，`prepare_dataset.py` 會檢查來源影像的 filename stem 是否唯一。

例如來源資料中若同時存在：

```text
folder_a/1234.jpg
folder_b/1234.png
```

兩張影像的 filename stem 都是：

```text
1234
```

此時資料配對具有歧義。

`prepare_dataset.py` 不會自動選擇其中一張影像，而是直接停止執行並回報：

```text
ValueError: Duplicate image filename stems found.
Each label stem must map to exactly one source image.
```

錯誤訊息後方會列出發生衝突的 filename stem 與所有對應檔案路徑。

此檢查可避免資料整理程式在沒有警告的情況下將錯誤來源影像配對至 YOLO Label。

---

## Preparing Images from Existing Labels

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

來源資料夾的子目錄也會被遞迴搜尋。

---

## What prepare_dataset.py Does

`prepare_dataset.py` 的用途是依照既有 Label 檔名整理影像，不負責建立 annotation。

程式流程為：

1. 遞迴掃描來源影像資料夾中的支援影像格式。
2. 以 filename stem 建立來源影像索引。
3. 檢查來源影像是否存在重複 filename stem。
4. 若同一 stem 對應多張來源影像，停止執行並列出所有衝突路徑。
5. 讀取 `labels/train`、`labels/val`、`labels/test` 中既有的 `.txt` 標註檔。
6. 取得每個標註檔的 filename stem。
7. 依照 filename stem 尋找對應來源影像。
8. 將找到的影像複製到對應的 `images/train`、`images/val` 或 `images/test`。
9. 記錄找不到對應影像的標註項目。

例如：

```text
labels/train/1234.txt
```

程式可搜尋：

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

## Supported Image Formats

目前 `prepare_dataset.py` 支援以下影像副檔名：

```text
.jpg
.jpeg
.png
.bmp
.webp
```

副檔名比對不區分大小寫。

---

## Missing Image Report

若某個 Label 找不到對應來源影像，程式會建立：

```text
missing_images.tsv
```

檔案會逐行記錄缺失項目的 split 與 filename stem。

例如：

```text
train	1234
val	5678_1
```

檔案目前不包含 header。

程式同時會在終端輸出：

```text
Copied images: ...
Missing images: ...
```

若存在缺失影像，也會顯示 `missing_images.tsv` 的位置。

常見原因包括：

* 影像檔名與 Label 檔名不一致
* 影像未放入指定來源資料夾
* Label 被保留，但原始影像已刪除
* 檔名中存在額外字元
* Split 整理錯誤

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

## Before Training

開始 YOLOv8 訓練前，應確認以下事項：

* `images/train` 與 `labels/train` 的資料可一一對應。
* `images/val` 與 `labels/val` 的資料可一一對應。
* `images/test` 與 `labels/test` 的資料可一一對應。
* Image 與 Label 的 filename stem 完全一致。
* 每個來源影像的 filename stem 具有唯一性。
* Label 內容符合 YOLO 格式。
* Bounding Box 座標已正規化至 0–1。
* `class_id` 僅使用已定義的類別。
* `data.yaml` 的資料集路徑正確。
* `data.yaml` 的類別順序與 annotation 一致。
* `missing_images.tsv` 中沒有尚未處理的必要資料。

---

## Dataset Used in This Project

本次實驗資料量為：

```text
Train:       1,401
Validation:    300
Test:          301
------------------
Total:        2,002
```

對應 YOLO Label 數量亦為：

```text
Train:       1,401
Validation:    300
Test:          301
------------------
Total:        2,002
```

本 repository 不公開完整資料集內容，但保留資料格式、設定範例、資料整理程式與實驗成果，以利理解整體專案架構與執行流程。
