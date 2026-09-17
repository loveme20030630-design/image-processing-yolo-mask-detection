# YOLO-based Mask Detection

以 YOLOv8 建立口罩佩戴狀態偵測模型，辨識 `with_mask` 與 `without_mask` 兩類目標。本專案源自數位影像處理課程實作，內容涵蓋資料整理、YOLO 格式轉換、模型訓練、驗證與測試、影像推論，以及 IoU／NMS 原理實作。

![口罩偵測範例](assets/sample_prediction.jpg)

## 專案重點

- 整理 4,193 張影像，切分為訓練、驗證與測試資料。
- 以相同資料與 50 epochs 設定比較 YOLOv8n 與 YOLOv8s。
- 使用 Precision、Recall、mAP@50 與 mAP@50–95 評估模型。
- 以獨立 Python 程式整理訓練、評估與推論流程，移除原始 Colab 的硬編碼路徑。
- 另行實作 IoU 與簡化版 Non-Maximum Suppression，理解偵測框篩選機制。

## 資料集

| 子集合 | with_mask | without_mask | 合計 |
|---|---:|---:|---:|
| Train | 1,400 | 1,535 | 2,935 |
| Validation | 300 | 328 | 628 |
| Test | 300 | 330 | 630 |
| **總計** | **2,000** | **2,193** | **4,193** |

資料集與模型權重未放入 repository：原始影像體積較大，且應分別確認資料來源授權。詳細目錄格式見 [`dataset/README.md`](dataset/README.md)。

## 實驗設定

| 項目 | 設定 |
|---|---|
| Framework | Ultralytics YOLOv8 |
| Models | YOLOv8n、YOLOv8s |
| Epochs | 50 |
| Image size | 640 × 640 |
| Batch size | 32 |
| Classes | `without_mask`、`with_mask` |
| Hardware | Google Colab GPU（可用時） |

## 實驗結果

以下數值來自保存的 `best.pt`，分別在 validation 與 test split 重新評估。

| Model | Split | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---|---:|---:|---:|---:|
| YOLOv8n | Validation | 0.977 | 0.954 | 0.972 | 0.695 |
| YOLOv8n | Test | 0.969 | 0.957 | 0.967 | 0.657 |
| YOLOv8s | Validation | 0.977 | 0.959 | 0.977 | 0.694 |
| YOLOv8s | Test | 0.959 | 0.975 | 0.978 | 0.657 |

YOLOv8s 的 test mAP@50 與 recall 較高，但兩者的 mAP@50–95 幾乎相同；因此不能只因模型較大就斷言效果明顯較好。若要實際部署，下一步應加入推論延遲、模型大小與裝置資源的比較。

### 訓練曲線

![YOLOv8s 訓練曲線](assets/results_yolov8s.png)

### 混淆矩陣

![YOLOv8s 混淆矩陣](assets/confusion_matrix_yolov8s.png)

### Validation 預測結果

![YOLOv8s validation predictions](assets/val_predictions_yolov8s.jpg)

## Repository 結構

```text
.
├── assets/                 # README 使用的成果圖
├── config/
│   └── data.example.yaml   # YOLO 資料集設定範例
├── dataset/
│   └── README.md           # 資料格式與類別說明
├── src/
│   ├── evaluate.py         # Validation / test 評估
│   ├── nms_demo.py         # IoU 與簡化版 NMS
│   ├── predict.py          # 圖片、資料夾或影片推論
│   ├── prepare_dataset.py  # 依標註檔補齊各 split 的影像
│   └── train.py            # YOLOv8 訓練入口
├── .gitignore
└── requirements.txt
```

## 快速開始

### 1. 安裝環境

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. 準備資料

依照 [`dataset/README.md`](dataset/README.md) 建立 YOLO 格式資料夾，複製 `config/data.example.yaml` 並修改其中的 `path`。

### 3. 訓練

```bash
python src/train.py --data config/data.yaml --model yolov8s.pt --epochs 50
```

若顯示記憶體不足，可先將 `--batch 32` 改為 `--batch 16` 或更小。

### 4. 評估

```bash
python src/evaluate.py --weights runs/mask_detection/yolov8s/weights/best.pt --data config/data.yaml --split val
python src/evaluate.py --weights runs/mask_detection/yolov8s/weights/best.pt --data config/data.yaml --split test
```

### 5. 推論

```bash
python src/predict.py --weights runs/mask_detection/yolov8s/weights/best.pt --source path/to/image_or_video
```

## 已知限制與後續方向

- 現有測試集與訓練資料來源相近，跨場景泛化能力仍需用不同來源影像驗證。
- `with_mask`／`without_mask` 只涵蓋兩類，未獨立處理口罩配戴不正確的情況。
- 目前 repository 不提供即時多人追蹤；Drive 中對應輸出資料夾為空，不能視為已完成成果。
- 後續可補充攝影機即時推論、邊緣裝置效能與誤判案例分析。

## 技術工具

Python、PyTorch、Ultralytics YOLOv8、PyYAML、Google Colab

