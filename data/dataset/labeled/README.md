# Labeled Dataset

此目錄包含已確認標記的訓練資料集，直接用於 ResNet50 模型訓練。

## 目錄說明

- `0/` ~ `9/`: 按數字標籤分類的圖像目錄
- 檔案命名格式: `{label}_{序號三位數}.png`
- 圖像規格: 224x224 RGB PNG 格式

## 使用方式

此目錄結構直接支援 TensorFlow/Keras 的內建函式：
- `tf.keras.utils.image_dataset_from_directory()`
- 支援 `validation_split` 參數自動分割訓練/驗證集

## 資料品質要求

- 圖像清晰，數字可辨識
- 正確的數字標籤分類
- 無損壞或模糊圖像
