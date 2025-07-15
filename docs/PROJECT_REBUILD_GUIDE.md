# Make10 遊戲自動化系統 - 專案重建指南

## 📋 專案概述

Make10 遊戲自動化系統是一個基於電腦視覺和機器學習的自動化工具，能夠自動檢測、分析並解決 Make10 數字拼圖遊戲。

## 🏗️ 系統架構與技術棧

### 核心技術
- **電腦視覺**: OpenCV (霍夫直線檢測、投影直方圖、模板匹配)
- **深度學習**: TensorFlow (數字識別模型)
- **圖像處理**: PIL/Pillow, NumPy
- **自動化控制**: PyInput (滑鼠控制), MSS (螢幕擷取)
- **數據分析**: Pandas
- **測試框架**: Pytest

### 專案結構
```
make10/
├── main.py                  # 主程式入口
├── src/                     # 核心程式碼
│   ├── ai/                  # AI 模組
│   ├── automation/          # 自動化模組 
│   ├── core/                # 核心邏輯
│   └── labeling/            # 標註工具
├── config/                  # 配置設定
├── data/                    # 數據和模型
└── tests/                   # 測試檔案
```

## 🔧 分階段實施計畫

### 階段 1: 基礎設施建置 (1-2 天)
**技術重點**: 專案結構、配置管理、基礎工具

1. **環境設定**
   - Python 3.8+ 虛擬環境
   - 套件相依性管理 (requirements.txt)
   - 開發工具設定 (ruff, pytest)

2. **配置系統** (`config/`)
   - `settings.py`: 路徑、參數配置
   - `constants.py`: 遊戲常數定義
   - 驗證機制和預設值

3. **基本結構建立**
   - 目錄結構建立
   - `__init__.py` 檔案
   - 基礎匯入路徑設定

### 階段 2: 電腦視覺模組 (3-4 天)
**技術重點**: OpenCV、圖像處理、座標檢測

1. **螢幕擷取** (`src/automation/screen_capture.py`)
   - MSS 螢幕擷取
   - 多螢幕支援
   - 區域擷取功能

2. **影像處理** (`src/ai/image_processor.py`)
   - OpenCV 預處理
   - 圖像正規化
   - 雜訊移除

3. **座標檢測** (`src/core/board_coordinate_detector.py`)
   - **霍夫直線檢測**: 自動偵測遊戲盤面網格線
   - **投影直方圖**: 輔助驗證網格線位置
   - **模板匹配**: 重置按鈕檢測
   - 動態座標計算 (取代固定位移)

### 階段 3: AI 識別系統 (2-3 天)
**技術重點**: TensorFlow、深度學習、預測模型

1. **模型管理** (`src/ai/model_manager.py`)
   - TensorFlow 模型載入
   - 模型版本管理
   - 記憶體優化

2. **預測器** (`src/ai/predictor.py`)
   - 單一細胞數字識別
   - 批次預測處理
   - 置信度評估
   - 投票機制提升準確率

3. **盤面掃描** (`src/core/board_scanner.py`)
   - 螢幕切換整合
   - 細胞圖像提取
   - AI 預測整合
   - 結果驗證

### 階段 4: 遊戲邏輯與求解 (2-3 天)
**技術重點**: 演算法、圖論、最佳化

1. **求解演算法** (`src/core/solver.py`)
   - **基礎求解**: 遞迴搜尋
   - **分支限界法**: 優化搜尋空間
   - **遺傳演算法**: 複雜情況處理
   - 多解答找尋

2. **遊戲引擎** (`src/core/game_engine.py`)
   - 統籌所有模組
   - 自動化流程控制
   - 狀態管理
   - 錯誤處理機制

### 階段 5: 自動化控制 (1-2 天)
**技術重點**: 滑鼠控制、動作序列、時間控制

1. **滑鼠控制** (`src/automation/mouse_control.py`)
   - PyInput 滑鼠操作
   - 精確座標計算
   - 拖拽動作實現
   - 延遲時間控制

2. **螢幕切換** (`src/automation/screen_utils.py`)
   - 自動螢幕切換
   - 視窗焦點管理
   - 操作穩定性保證

### 階段 6: 資料管理與標註 (2 天)
**技術重點**: 資料處理、人工標註、品質控制

1. **標籤管理** (`src/labeling/label_manager.py`)
   - CSV 資料管理
   - 標籤統計分析
   - 備份和驗證

2. **標註工具** (`src/labeling/annotation_tool.py`)
   - 互動式標註介面
   - 快捷鍵操作
   - 品質檢查機制

### 階段 7: 測試與整合 (2-3 天)
**技術重點**: 單元測試、整合測試、效能監控

1. **階段化測試系統**
   - `test_initialization.py`: 組件初始化測試
   - `test_game_detection.py`: 遊戲偵測測試  
   - `test_board_scanning.py`: 盤面掃描測試
   - `test_screen_switching.py`: 螢幕切換測試
   - `test_full_integration.py`: 端到端測試

2. **效能監控** (`tests/performance_monitor.py`)
   - 記憶體使用追蹤
   - 執行時間分析
   - 系統資源監控

## 🛠️ 關鍵技術實現

### 1. 動態座標檢測 (核心技術)
```python
# 霍夫直線檢測 + 投影直方圖
def detect_board_coordinates(image):
    # 1. 圖像預處理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # 2. 霍夫直線檢測
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100)
    
    # 3. 投影直方圖驗證
    h_projection = np.sum(edges, axis=1)
    v_projection = np.sum(edges, axis=0)
    
    # 4. 網格線分類和座標計算
    return calculate_grid_coordinates(lines, h_projection, v_projection)
```

### 2. 螢幕切換整合流程
```python
def scan_board_with_screen_switch():
    # 1. 切換到遊戲螢幕
    switch_screen()
    time.sleep(0.5)
    
    # 2. 執行盤面掃描
    board, confidence = scan_board()
    
    # 3. 切換回原始螢幕
    switch_screen()
    time.sleep(0.3)
    
    return board, confidence
```

### 3. AI 預測管道
```python
# 端到端預測流程
def predict_board_numbers(cell_images):
    # 1. 圖像預處理
    processed_cells = [preprocess_cell(img) for img in cell_images]
    
    # 2. 批次預測
    predictions = model.predict(np.array(processed_cells))
    
    # 3. 置信度評估和投票機制
    final_results = apply_voting_mechanism(predictions)
    
    return final_results
```

## 📦 套件需求

詳見更新的 `requirements.txt`，包含所有必要的 Python 套件和版本要求。

## 🚀 快速開始步驟

1. **環境準備**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **基礎測試**
   ```bash
   python tests/test_initialization.py
   python tests/test_game_detection.py
   ```

3. **完整執行**
   ```bash
   python main.py  # 主程式
   python src/main.py  # 替代入口
   ```

## ⚠️ 實施注意事項

1. **開發順序**: 嚴格按照階段順序，確保基礎穩固
2. **測試驅動**: 每個階段完成後進行對應測試
3. **模組獨立**: 確保各模組可獨立測試和使用
4. **錯誤處理**: 重視異常處理和錯誤恢復機制
5. **效能優化**: 監控記憶體使用和執行時間

## 📈 預期成果

- **準確率**: 數字識別準確率 85%+
- **效能**: 完整掃描 < 2 秒
- **穩定性**: 連續執行 100+ 回合無錯誤
- **維護性**: 模組化設計，易於擴展和維護

此指南提供了完整的重建藍圖，按照此計畫可以在 2-3 週內重建一個功能完整、架構清晰的 Make10 自動化系統。
