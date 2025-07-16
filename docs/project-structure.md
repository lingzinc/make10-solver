# 專案結構文件

> 📁 Make10 專案的目錄結構、模組組織與檔案功能說明

## 🗂️ 專案架構總覽

```
make10-solver/
├── � pyproject.toml          # 專案配置與依賴管理 (UV/pip)
├── � pytest.ini             # 測試框架配置
├── � README.md              # 專案首頁說明
├── � uv.lock                # 依賴鎖定檔案 (UV 自動生成)
├── �🚀 run_system.py          # 系統啟動入口
├── 🧠 run_training.py        # AI 訓練入口 (開發中)
│
├── ⚙️ config/                # 配置管理系統
│   ├── __init__.py           # 模組初始化
│   ├── constants.py          # 系統常數定義
│   └── settings.py           # 配置物件與路徑管理
│
├── 📊 data/                  # 資料存儲目錄
│   ├── DATA_STRUCTURE.md     # 資料結構說明
│   ├── assets/               # 靜態資源
│   │   └── templates/        # 影像模板檔案
│   ├── models/               # AI 模型存儲
│   │   ├── checkpoints/      # 訓練檢查點
│   │   └── exports/          # 匯出模型
│   └── training/             # 訓練資料集
│       ├── images/           # 訓練影像
│       └── labels/           # 標籤資料
│
├── 📚 docs/                  # 技術文件集
│   ├── README.md             # 文件總覽
│   ├── development-guide.md  # 開發指南
│   ├── technical-architecture.md # 技術架構
│   ├── installation.md      # 安裝指南
│   └── ...                  # 其他專業文件
│
├── 📝 logs/                  # 系統執行日誌
│   └── make10_system.log     # 主要日誌檔案
│
├── 🧩 src/                   # 核心程式碼模組
│   ├── __init__.py
│   ├── ai/                   # AI 與機器學習 (規劃中)
│   │   └── __init__.py
│   ├── automation/           # 自動化控制模組 ✅
│   │   ├── __init__.py
│   │   ├── keyboard_listener.py  # 鍵盤監聽與熱鍵
│   │   └── screen_utils.py       # 螢幕操作工具
│   ├── core/                 # 核心系統邏輯 ✅
│   │   ├── __init__.py
│   │   └── main.py           # 主要業務邏輯
│   └── labeling/             # 資料標籤工具 (規劃中)
│       └── __init__.py
│
└── 🧪 tests/                 # 測試套件 ✅
    ├── __init__.py
    ├── test_config_settings.py    # 配置系統測試
    ├── test_keyboard_listener.py  # 鍵盤監聽測試
    └── test_screen_utils.py       # 螢幕工具測試
```

## 📋 核心檔案詳解

### 🚀 入口檔案

#### `run_system.py` - 系統主入口
```python
"""Make10 遊戲自動化系統啟動器"""
from src.core.main import main

# 功能:
# - 系統初始化與日誌設定
# - 依賴套件檢查
# - 主程式啟動與錯誤處理
# - 安全退出機制
```

#### `run_training.py` - AI 訓練入口
```python
"""AI 模型訓練系統 (開發中)"""
# 預計功能:
# - 載入訓練資料集
# - 模型架構初始化
# - 訓練流程執行
# - 模型評估與儲存
```

### ⚙️ 配置系統

#### `config/settings.py` - 配置管理核心
```python
from easydict import EasyDict
cfg = EasyDict()

# 路徑配置
cfg.PATHS.MODEL.main_model = "data/models/exports/model.keras"
cfg.PATHS.TRAINING.images_dir = "data/training/images"

# 系統參數
cfg.AUTOMATION.click_delay = 0.1
cfg.MODEL.confidence_threshold = 0.8
cfg.SYSTEM.exit_key = "ctrl+q"
```

#### `config/constants.py` - 系統常數
```python
# 影像處理常數
CELL_SIZE = 50
BOARD_SIZE = (4, 4)
CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 150

# 自動化控制常數
CLICK_DELAY = 0.1
SCREENSHOT_DELAY = 0.05
RETRY_ATTEMPTS = 3
```

### 🧩 核心模組

#### `src/core/main.py` - 系統核心
```python
class GameAutomationSystem:
    """Make10 遊戲自動化系統核心類別"""
    
    def __init__(self):
        self.initialized = False
        self.keyboard_listener = None
        self.should_exit = False
    
    def initialize(self):
        """初始化系統核心模組"""
        # 初始化鍵盤監聽器
        # TODO: 初始化 AI 模型
        # TODO: 初始化電腦視覺模組
    
    def run_game_loop(self):
        """執行主要遊戲自動化流程"""
        # 遊戲邏輯實作
```

#### `src/automation/screen_utils.py` - 螢幕操作
```python
# 主要功能函式:
def capture_screen() -> np.ndarray          # 螢幕擷取
def click_at_position(x, y) -> bool         # 精確點擊
def find_reset_button() -> tuple           # 模板匹配
def switch_screen() -> bool                # 視窗切換 (Alt+Tab)
```

#### `src/automation/keyboard_listener.py` - 鍵盤監聽
```python
def create_keyboard_listener(exit_callback):
    """建立鍵盤監聽器"""
    # 熱鍵組合監聽
    # 安全退出機制
    # 背景執行支援
```

### 🧪 測試系統

#### 測試覆蓋模組
```python
# tests/test_screen_utils.py
def test_switch_screen_success()           # 螢幕切換測試
def test_capture_screen()                  # 螢幕擷取測試
def test_click_at_position()               # 滑鼠點擊測試

# tests/test_keyboard_listener.py  
def test_keyboard_listener_creation()      # 鍵盤監聽器測試

# tests/test_config_settings.py
def test_config_loading()                  # 配置載入測試
def test_path_configuration()              # 路徑配置測試
```

## 📊 模組狀態與開發進度

### ✅ 已完成模組
| 模組 | 檔案 | 功能狀態 | 測試覆蓋 |
|------|------|----------|----------|
| **核心系統** | `src/core/main.py` | ✅ 完成 | ✅ 有測試 |
| **螢幕工具** | `src/automation/screen_utils.py` | ✅ 完成 | ✅ 有測試 |
| **鍵盤監聽** | `src/automation/keyboard_listener.py` | ✅ 完成 | ✅ 有測試 |
| **配置系統** | `config/settings.py` | ✅ 完成 | ✅ 有測試 |

### 🚧 開發中模組
| 模組 | 規劃功能 | 開發狀態 | 優先級 |
|------|----------|----------|--------|
| **AI 模型** | `src/ai/` | 🚧 規劃中 | 高 |
| **標籤工具** | `src/labeling/` | 🚧 規劃中 | 中 |
| **訓練系統** | `run_training.py` | 🚧 開發中 | 高 |

## 🔗 模組依賴關係

```mermaid
graph TD
    A[run_system.py] --> B[src/core/main.py]
    B --> C[src/automation/screen_utils.py]
    B --> D[src/automation/keyboard_listener.py]
    C --> E[config/settings.py]
    D --> E
    E --> F[config/constants.py]
    
    G[run_training.py] --> H[src/ai/ (規劃中)]
    H --> I[src/labeling/ (規劃中)]
    H --> E
    
    J[tests/] --> B
    J --> C
    J --> D
    J --> E
```

## 📁 資料目錄結構

### `data/` 目錄組織
```
data/
├── assets/                    # 靜態資源
│   └── templates/            # OpenCV 模板匹配檔案
│       ├── reset_button_b.png  # 重置按鈕 (黑色主題)
│       └── reset_button_w.png  # 重置按鈕 (白色主題)
│
├── models/                   # AI 模型存儲
│   ├── checkpoints/         # 訓練過程檢查點
│   └── exports/             # 最終匯出模型
│       └── model.keras      # 主要 TensorFlow 模型
│
└── training/                # 訓練資料集
    ├── images/              # 訓練影像檔案
    └── labels/              # 對應標籤資料
```

## 🛠️ 開發工作流程

### 新增功能模組步驟
1. **在 `src/` 下建立模組目錄**
2. **撰寫核心功能程式碼**
3. **更新 `config/settings.py` 相關配置**
4. **在 `tests/` 建立對應測試檔案**
5. **更新此文件的模組說明**

### 檔案命名規範
- **模組檔案**: `module_name.py` (小寫加底線)
- **類別檔案**: `ClassName.py` (大駝峰命名)
- **測試檔案**: `test_module_name.py` (test_ 前綴)
- **配置檔案**: `settings.py`, `constants.py` (描述性命名)

## 📈 專案擴展計畫

### 短期目標 (1-2 個月)
- [ ] 完成 `src/ai/` 模組開發
- [ ] 實作 `run_training.py` 訓練流程
- [ ] 增加更多自動化測試

### 中期目標 (3-6 個月)  
- [ ] 開發 `src/labeling/` 標籤工具
- [ ] 建立 CI/CD 自動化流程
- [ ] 效能最佳化與記憶體管理

### 長期目標 (6+ 個月)
- [ ] 圖形使用者介面 (GUI)
- [ ] 多遊戲支援架構
- [ ] 雲端部署與 API 服務

#### `pytest.ini`
```ini
# 測試框架配置
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    unit: 標記為單元測試
    integration: 標記為整合測試
    slow: 標記為慢速測試
```

### 主要執行檔案

#### `run_system.py`
```python
"""
系統主程式入口
- 載入系統配置
- 初始化所有模組
- 執行主要自動化流程
- 處理系統級異常
"""
from src.core.main import main

if __name__ == "__main__":
    main()
```

#### `run_training.py`
```python
"""
AI 模型訓練程式
- 資料載入與預處理
- 模型訓練與驗證
- 模型儲存與部署
- 支援增量訓練
"""
```

## ⚙️ config/ - 配置管理模組

### 目錄結構
```
config/
├── __init__.py           # 模組初始化
├── constants.py          # 系統常數定義
├── settings.py           # 配置設定管理
└── __pycache__/         # Python 快取檔案
```

### 檔案說明

#### `constants.py`
```python
"""
系統常數定義
- 遊戲相關常數 (盤面大小、數字範圍)
- 圖像處理參數 (尺寸、閾值)
- 演算法參數 (迭代次數、容錯度)
- 硬體相關常數 (螢幕解析度、DPI)
"""

# 遊戲常數
BOARD_SIZE = (5, 5)          # 盤面大小
CELL_SIZE = (45, 45)         # 單元格大小
TARGET_SUM = 10              # 目標總和

# AI 模型常數
MODEL_INPUT_SIZE = (28, 28)  # 模型輸入尺寸
NUM_CLASSES = 10             # 分類數量
CONFIDENCE_THRESHOLD = 0.8   # 信心度門檻
```

#### `settings.py`
```python
"""
系統設定管理
- 路徑配置 (模型路徑、資料路徑)
- 執行參數 (延遲時間、重試次數)
- 除錯設定 (日誌等級、輸出選項)
- 環境變數處理
"""

class Settings:
    def __init__(self):
        self.model_path = self._get_model_path()
        self.data_path = self._get_data_path()
        self.log_level = self._get_log_level()
        # ... 其他設定
```

## 📊 data/ - 資料存儲目錄

### 目錄結構
```
data/
├── DATA_STRUCTURE.md         # 資料結構說明
├── assets/                   # 靜態資源
│   └── templates/           # 模板圖像
│       ├── reset_button_b.png  # 重置按鈕 (黑色主題)
│       └── reset_button_w.png  # 重置按鈕 (白色主題)
├── models/                   # AI 模型存儲
│   ├── checkpoints/         # 訓練檢查點
│   └── exports/             # 導出模型
│       └── model.keras      # 主要模型檔案
└── training/                # 訓練資料
    ├── images/              # 訓練圖像
    └── labels/              # 標籤檔案
```

### 資料管理說明

#### 模型檔案管理
```python
# 模型檔案命名規範
model.keras                   # 當前使用模型
model_backup_YYYYMMDD.keras  # 按日期備份
model_v1.0.0.keras           # 版本化模型
```

#### 訓練資料組織
```python
# 圖像檔案命名
cell_001.png                 # 編號從 001 開始
cell_002.png                 # 連續編號

# 標籤檔案格式 (labels.txt)
cell_001.png,5              # 檔案名,對應數字
cell_002.png,3
```

## 🧩 src/ - 核心程式碼模組

### 目錄結構
```
src/
├── __init__.py              # 模組初始化
├── ai/                      # AI 相關模組
│   └── __init__.py
├── automation/              # 自動化控制模組
│   ├── __init__.py
│   ├── keyboard_listener.py # 鍵盤監聽器
│   ├── screen_utils.py      # 螢幕工具
│   └── __pycache__/
├── core/                    # 核心邏輯模組
│   ├── __init__.py
│   ├── main.py              # 主要邏輯
│   └── __pycache__/
├── labeling/                # 資料標註模組
│   └── __init__.py
└── __pycache__/
```

### 模組詳細說明

#### `src/ai/` - AI 相關模組
```python
# 預期包含的檔案
model_manager.py             # AI 模型管理
├── load_model()            # 載入模型
├── save_model()            # 儲存模型
└── get_model_info()        # 模型資訊

predictor.py                 # 預測介面
├── predict_single_cell()   # 單一預測
├── predict_batch_cells()   # 批次預測
└── predict_with_confidence() # 含信心度預測

image_processor.py           # 圖像預處理
├── preprocess_cell()       # Cell 預處理
├── normalize_image()       # 圖像正規化
└── augment_data()          # 資料增強
```

#### `src/automation/` - 自動化控制模組

##### `keyboard_listener.py`
```python
"""
鍵盤監聽器
- 熱鍵註冊與處理
- 系統快捷鍵監聽
- 程式啟動/停止控制
- 安全退出機制
"""

class KeyboardListener:
    def __init__(self):
        self.is_running = False
        self.hotkeys = {}
    
    def register_hotkey(self, key_combination, callback):
        """註冊熱鍵回調"""
    
    def start_listening(self):
        """開始監聽鍵盤事件"""
    
    def stop_listening(self):
        """停止監聽"""
```

##### `screen_utils.py`
```python
"""
螢幕工具
- 螢幕擷取功能
- 多螢幕支援
- 視窗定位與切換
- 座標系統轉換
"""

class ScreenUtils:
    def capture_screen(self, region=None):
        """擷取螢幕區域"""
    
    def get_screen_count(self):
        """取得螢幕數量"""
    
    def switch_to_screen(self, screen_index):
        """切換到指定螢幕"""
    
    def find_window_by_title(self, title):
        """根據標題尋找視窗"""
```

#### `src/core/` - 核心邏輯模組

##### `main.py`
```python
"""
系統主要邏輯
- 系統初始化
- 主執行循環
- 模組協調
- 異常處理
"""

def main():
    """主程式入口"""
    try:
        system = initialize_system()
        run_main_loop(system)
    except KeyboardInterrupt:
        logger.info("使用者中斷程式執行")
    except Exception as e:
        logger.error(f"系統異常: {e}")
    finally:
        cleanup_system()

def initialize_system():
    """初始化系統所有模組"""
    
def run_main_loop(system):
    """主要執行循環"""
```

#### `src/labeling/` - 資料標註模組
```python
# 預期包含的檔案
annotation_tool.py           # 標註工具
├── display_image()         # 顯示圖像
├── get_user_input()        # 取得使用者輸入
└── save_label()            # 儲存標籤

label_manager.py             # 標籤管理
├── load_labels()           # 載入標籤
├── save_labels()           # 儲存標籤
├── validate_labels()       # 驗證標籤
└── export_training_data()  # 導出訓練資料
```

## 🧪 tests/ - 測試檔案模組

### 目錄結構
```
tests/
├── __init__.py                      # 測試模組初始化
├── test_config_settings.py         # 配置設定測試
├── test_keyboard_listener.py       # 鍵盤監聽器測試
├── test_screen_utils.py            # 螢幕工具測試
└── __pycache__/                    # 測試快取檔案
```

### 測試組織原則

#### 測試檔案命名規範
```python
test_<module_name>.py               # 對應模組的測試
test_integration_<feature>.py       # 整合測試
test_performance_<component>.py     # 效能測試
```

#### 測試類別組織
```python
# test_config_settings.py
class TestConfigurationObject:      # 配置物件測試
class TestPathValidation:          # 路徑驗證測試
class TestConfigurationValues:     # 配置值測試
class TestModelPath:               # 模型路徑測試
class TestDebugDirectory:          # 除錯目錄測試
class TestInitializeSettings:      # 初始化設定測試
```

## 📚 docs/ - 專案文件模組

### 重新整理後結構
```
docs/
├── README.md                       # 文件總覽
├── getting-started.md              # 快速入門
├── installation.md                 # 安裝指南
├── system-workflow.md              # 系統工作流程
├── development-guide.md            # 開發指南
├── project-structure.md           # 本檔案
├── ai-model-guide.md              # AI 模型指南
├── training-workflow.md           # 訓練工作流程
├── technical-architecture.md      # 技術架構
├── computer-vision.md             # 電腦視覺
├── testing-guide.md               # 測試指南
├── quality-assurance.md           # 品質保證
├── troubleshooting.md             # 故障排除
└── archive/                       # 原始文件備份
    ├── AI_MODEL_REBUILD_GUIDE.md
    ├── PRE_PUSH_SETUP.md
    ├── PROJECT_REBUILD_GUIDE.md
    ├── PYTEST_GUIDE.md
    ├── STEP.md
    └── TECH_SUMMARY.md
```

## 📝 logs/ - 執行日誌模組

### 日誌檔案組織
```
logs/
├── make10_system.log              # 系統主日誌
├── ai_model.log                   # AI 模型日誌
├── automation.log                 # 自動化日誌
├── error.log                      # 錯誤日誌
└── performance.log                # 效能監控日誌
```

### 日誌輪換策略
```python
# 日誌設定範例
LOGGING_CONFIG = {
    'version': 1,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        }
    },
    'handlers': {
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/make10_system.log',
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5,
            'formatter': 'detailed'
        }
    }
}
```

## 🔄 模組相依關係

### 相依關係圖
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   config    │────│    core     │────│ automation  │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   logging   │    │     ai      │    │   testing   │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    data     │    │  labeling   │    │    docs     │
└─────────────┘    └─────────────┘    └─────────────┘
```

### 匯入層次規則
```python
# 層次 1: 基礎模組 (無相依性)
from config import settings, constants

# 層次 2: 工具模組 (依賴基礎模組)
from src.automation import screen_utils, keyboard_listener

# 層次 3: 核心模組 (依賴基礎與工具模組)
from src.ai import model_manager, predictor
from src.core import main

# 避免循環相依
# ❌ 錯誤: core 模組不應匯入 automation 模組內的 core 相關功能
# ✅ 正確: 使用依賴注入或事件機制
```

## 📦 套件結構設計

### 模組介面設計
```python
# 每個模組的 __init__.py 應該暴露主要介面
# src/ai/__init__.py
from .model_manager import ModelManager
from .predictor import Predictor
from .image_processor import ImageProcessor

__all__ = ['ModelManager', 'Predictor', 'ImageProcessor']

# src/automation/__init__.py
from .screen_utils import ScreenUtils
from .keyboard_listener import KeyboardListener

__all__ = ['ScreenUtils', 'KeyboardListener']
```

### 版本管理
```python
# src/__init__.py
__version__ = "1.0.0"
__author__ = "Make10 Development Team"
__email__ = "dev@make10.com"

# 版本號規則: MAJOR.MINOR.PATCH
# MAJOR: 不相容的 API 變更
# MINOR: 向後相容的功能新增
# PATCH: 向後相容的錯誤修復
```

透過這個結構化的組織，Make10 專案保持良好的可維護性、可擴展性和可測試性。每個模組都有明確的職責，模組間的相依關係清晰，便於團隊協作開發。
