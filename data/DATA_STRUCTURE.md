# Make10 資料目錄架構

這個目錄包含 Make10 遊戲自動化系統所需的所有資料檔案，專為數字辨識模組設計。

## 目錄結構

### 📁 dataset/
數字辨識訓練資料集
- `labeled/` - 已標記的訓練資料集 (支援 Keras 內建分割)
  - `0/` ~ `9/` - 按數字標籤分類的圖像目錄
- `unlabeled/` - 待標記與處理的圖像
  - `pending_manual/` - 待人工標記
    - `low_confidence/` - 低信任度圖像 (<0.7)
    - `medium_confidence/` - 中信任度圖像 (0.7-0.9)
    - `review_queue/` - 待審核佇列
  - `auto_labeled/` - 自動標記結果
    - `high_confidence/` - 高信任度預測 (>0.9)
      - `predicted_0/` ~ `predicted_9/` - 按預測結果分類
    - `metadata/` - 預測相關資訊
  - `rejected/` - 品質不佳的圖像
    - `blurry/` - 模糊圖像
    - `corrupted/` - 損壞圖像
    - `invalid/` - 無效圖像
- `metadata/` - 資料集管理資訊
  - `dataset_info.json` - 資料集統計資訊
  - `labeling_log.json` - 標記記錄
  - `quality_report.json` - 品質檢查報告
  - `training_config.json` - 訓練設定

### 📁 assets/
靜態圖像資產
- `templates/` - 模板匹配用的參考圖像（按鈕圖示等）

### 📁 models/
ResNet50 機器學習模型檔案
- `checkpoints/` - 訓練過程中的模型檢查點
  - `stage1_best.keras` - 第一階段最佳模型
  - `stage2_best.keras` - 第二階段最佳模型
- `exports/` - 最終匯出的模型檔案
  - `resnet50_model.keras` - 主要的 ResNet50 模型檔案
