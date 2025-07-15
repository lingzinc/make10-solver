# Make10 AI 模型重建指南

## 📋 重建概覽

### 專案資訊
- **專案名稱**: Make10 遊戲自動化 AI 模型系統
- **模型類型**: CNN 數字識別模型 (TensorFlow/Keras)
- **目標**: 自動識別 10x25 遊戲網格中的數字 (0-9)
- **輸入格式**: 28x28 灰階圖像
- **輸出格式**: 10 類別 softmax 分類
- **建立時間**: 2025年7月15日

---

## 🎯 模型架構設計

### 模型結構 (CNN)
```python
model = Sequential([
    Input(shape=(28, 28, 1)),
    # 第一個卷積塊 - 簡化
    Conv2D(16, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    # 第二個卷積塊 - 簡化
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    # 全連接層 - 大幅簡化
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax"),
])
```

### 技術規格
- **模型類型**: Sequential CNN
- **輸入形狀**: (28, 28, 1) - 28x28 灰階圖像
- **卷積層**: 2層，濾波器數量 [16, 32]
- **池化層**: 2x2 MaxPooling
- **正規化**: Dropout [0.2, 0.3, 0.5]
- **全連接層**: 64 neurons
- **輸出層**: 10 類別 softmax
- **優化器**: Adam
- **損失函式**: categorical_crossentropy
- **評估指標**: accuracy

---

## 🔧 6 階段重建計畫

### 階段 1: 環境設定 (30分鐘)
**目標**: 建立完整的開發環境

#### 1.1 套件安裝
```powershell
# 建立虛擬環境
uv venv

# 啟動環境
.venv\Scripts\activate

# 安裝核心套件
uv add tensorflow>=2.13.0
uv add opencv-python>=4.8.0
uv add numpy>=1.24.0
uv add scikit-learn>=1.3.0
uv add matplotlib>=3.7.0
uv add pillow>=10.0.0
```

#### 1.2 目錄結構建立
```
make10/
├── data/
│   ├── models/              # 模型檔案存放
│   ├── raw/
│   │   └── cell_images/     # 原始訓練圖像
│   └── labels/              # 標籤檔案
│       └── labels.txt
└── src/
    └── ai/
        ├── model_manager.py # 模型管理
        ├── predictor.py     # 預測器
        └── image_processor.py # 圖像處理
```

### 階段 2: 資料準備與標註 (1-2 天)
**目標**: 準備高品質訓練資料

#### 2.1 資料收集
```python
# 從遊戲擷取 cell 圖像
def capture_game_cells():
    # 自動化擷取遊戲盤面
    # 分割成 250 個 45x45 的 cell
    # 儲存為 cell_{index}.png 格式
```

#### 2.2 手動標註
```powershell
# 啟動標註工具
uv run run_training.py --label

# 標註流程：
# - 顯示每個 cell 圖像
# - 人工輸入對應數字 (0-9)
# - 自動儲存到 labels.txt
```

#### 2.3 資料品質控制
- **最少樣本數**: 每個數字至少 50 個樣本
- **資料平衡**: 各數字分佈盡量均勻
- **品質檢查**: 剔除模糊、錯誤的圖像

### 階段 3: 模型訓練核心建構 (1 天)
**目標**: 實作訓練pipeline

#### 3.1 資料載入器 (整合於 `run_training.py`)
```python
def load_cell_images():
    """從 data/raw/cell_images/ 載入圖像和標籤"""
    # 讀取圖像檔案
    # 對應標籤檔案
    # 正規化處理 (0-1 範圍)
    # 調整大小為 28x28
    return images, labels
```

#### 3.2 模型建構函式
```python
def create_or_load_model(force_new=False):
    """建立新模型或載入現有模型"""
    # 檢查現有模型檔案
    # 支援增量訓練
    # 模型架構定義
    # 編譯設定
```

#### 3.3 訓練設定
```python
# 回調函式
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
]

# 訓練參數
epochs = 20
batch_size = 16
validation_split = 0.3
```

### 階段 4: 模型管理系統 (半天)
**目標**: 實作模型載入、儲存、管理

#### 4.1 模型管理器 (`src/ai/model_manager.py`)
```python
class ModelManager:
    def __init__(self, settings):
        self.settings = settings
        self._model = None
    
    def load_model(self) -> bool:
        """載入 TensorFlow 模型"""
        # 檢查檔案存在
        # 載入模型
        # 錯誤處理
    
    def save_model(self, model, filename=None) -> bool:
        """儲存模型"""
        # 建立目錄
        # 儲存模型
        # 確認儲存成功
    
    def model_exists(self) -> bool:
        """檢查模型檔案是否存在"""
    
    def get_model_info(self) -> dict:
        """取得模型資訊"""
```

### 階段 5: 預測系統實作 (1 天)
**目標**: 建立高效的預測介面

#### 5.1 圖像預處理 (`src/ai/image_processor.py`)
```python
def preprocess_cell(cell_img):
    """預處理單一 cell 圖像"""
    # 灰階轉換
    # 正規化
    # 調整大小
    # 去雜訊
    return processed_img

def prepare_batch_input(cells_data):
    """準備批次輸入"""
    # 格式轉換
    # 維度調整
    return batch_input
```

#### 5.2 預測器 (`src/ai/predictor.py`)
```python
class Predictor:
    def predict_single_cell(self, cell_data) -> int:
        """預測單一 cell"""
    
    def predict_batch_cells(self, cells_data) -> np.ndarray:
        """批次預測"""
    
    def predict_with_confidence(self, cells_data):
        """預測並返回信心度"""
    
    def predict_with_voting(self, cell) -> int:
        """投票機制提升準確度"""
    
    def analyze_board_predictions(self, predictions, confidence):
        """分析預測結果"""
```

### 階段 6: 整合與最佳化 (1 天)
**目標**: 系統整合與效能最佳化

#### 6.1 系統整合
```python
# 主程式整合 (main.py)
def get_model():
    """單例模式載入模型"""
    global _model
    if _model is None:
        _model = tf.keras.models.load_model("model.keras")
    return _model

def scanBoard(img, init_x, init_y):
    """掃描遊戲盤面"""
    model = get_model()
    # 批次處理 250 個 cell
    # 一次性預測
    # 信心度評估
    # 低信心度處理
```

#### 6.2 效能最佳化
- **批次預測**: 一次處理 250 個 cell
- **記憶體管理**: 單例模式載入模型
- **快取機制**: 避免重複計算
- **並行處理**: 圖像預處理並行化

#### 6.3 品質保證
```python
def save_low_confidence_cells_with_manual_labeling():
    """處理低信心度預測"""
    # 自動儲存問題 cell
    # 即時人工標註
    # 累積重新訓練資料
```

---

## 🛠️ 實作細節

### 訓練程式執行
```powershell
# 全新模型訓練
uv run run_training.py --new

# 增量訓練
uv run run_training.py

# 使用 MNIST 備選資料集
uv run run_training.py --mnist
```

### 模型檔案管理
- **主模型**: `data/models/model.keras`
- **備份模型**: 自動時間戳備份
- **模型資訊**: 輸入形狀、參數數量、層數

### 資料管理
- **訓練圖像**: `data/raw/cell_images/cell_*.png`
- **標籤檔案**: `data/labels/labels.txt` (格式: filename,label)
- **品質控制**: 自動檢測並排除問題資料

---

## 🔬 技術實作要點

### 圖像預處理流程
1. **擷取**: 從遊戲畫面擷取 45x45 cell
2. **調整**: resize 到 28x28
3. **正規化**: 像素值 0-1
4. **增強**: 可選的資料增強

### 模型訓練策略
1. **基礎訓練**: 使用標註資料或 MNIST
2. **增量訓練**: 基於新的低信心度資料
3. **遷移學習**: 從 MNIST 遷移到遊戲資料
4. **正規化**: Dropout 防止過擬合

### 預測最佳化
1. **批次處理**: 250 個 cell 一次預測
2. **信心度評估**: 識別需要重新標註的 cell
3. **投票機制**: 多種預處理方法投票
4. **自動標註**: 低信心度 cell 即時標註

---

## 📊 品質指標

### 模型效能目標
- **準確率**: >95% (測試集)
- **信心度**: >0.8 (高品質預測)
- **推理速度**: <100ms (250 cells)
- **記憶體使用**: <500MB

### 資料品質要求
- **最少樣本**: 每類別 50+ 樣本
- **標註準確率**: >99%
- **圖像品質**: 清晰、無噪音
- **資料平衡**: 各類別比例相近

---

## 🔄 持續改進流程

### 增量學習流程
1. **執行主程式** → 自動識別低信心度 cell
2. **即時標註** → 人工修正錯誤預測
3. **累積資料** → 儲存新標註資料
4. **重新訓練** → `uv run run_training.py`
5. **模型更新** → 自動載入新模型

### 模型版本管理
- **自動備份**: 訓練前備份舊模型
- **效能監控**: 追蹤準確率變化
- **回滾機制**: 效能下降時恢復舊版本

---

## 🚀 一鍵執行指南

### 快速開始
```powershell
# 1. 環境準備
uv venv && .venv\Scripts\activate && uv sync

# 2. 建立目錄
mkdir -p data/models data/raw/cell_images data/labels

# 3. 準備資料 (手動擷取遊戲 cell 或使用 MNIST)
uv run run_training.py --label

# 4. 訓練模型
uv run run_training.py --new

# 5. 整合測試
uv run run_system.py
```

### 故障排除
- **TensorFlow 安裝問題**: 檢查 Python 版本相容性
- **記憶體不足**: 減少 batch_size
- **訓練資料不足**: 使用 MNIST 備選
- **模型不收斂**: 調整學習率或網路結構

---

## 📈 預期時程

| 階段 | 時間 | 關鍵里程碑 |
|------|------|------------|
| 階段1 | 30分鐘 | 環境就緒 |
| 階段2 | 1-2天 | 資料標註完成 |
| 階段3 | 1天 | 訓練pipeline就緒 |
| 階段4 | 半天 | 模型管理完成 |
| 階段5 | 1天 | 預測系統完成 |
| 階段6 | 1天 | 系統整合完成 |
| **總計** | **3-5天** | **完整AI模型系統** |

---

## 🎯 成功標準

✅ **模型訓練成功** - 達到 >90% 測試準確率  
✅ **預測系統穩定** - 批次預測無錯誤  
✅ **整合流暢** - 與主程式無縫整合  
✅ **增量學習** - 支援持續改進  
✅ **效能達標** - 滿足即時預測需求

重建後的 AI 模型系統將具備完整的訓練、預測、管理和持續改進能力！
