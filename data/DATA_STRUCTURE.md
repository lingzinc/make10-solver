# Make10 資料目錄

這個目錄包含 Make10 遊戲自動化系統所需的所有資料檔案。

## 目錄結構

### 📁 training/
訓練集相關資料
- `images/` - 訓練用的遊戲截圖和圖像資料
- `labels/` - 對應的標籤檔案（數字識別、遊戲狀態等）

### 📁 assets/
靜態圖像資產
- `templates/` - 模板匹配用的參考圖像（數字模板、按鈕圖示等）

### 📁 models/
機器學習模型檔案
- `checkpoints/` - 訓練過程中的模型檢查點
- `exports/` - 最終匯出的模型檔案
  - `model.keras` - 主要的 Keras 模型檔案
