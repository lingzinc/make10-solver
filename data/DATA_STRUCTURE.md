# Make10 資料目錄 (ResNet50 版本)

這個目錄包含 Make10 遊戲自動化系統所需的所有資料檔案，專為 ResNet50 模型最佳化。

## 目錄結構

### 📁 training/
ResNet50 訓練集相關資料
- `images/` - 訓練用的高解析度遊戲截圖 (224x224 RGB)
- `labels/` - 對應的標籤檔案（數字識別 0-9）

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
