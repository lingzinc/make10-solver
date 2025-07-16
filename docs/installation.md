# 環境安裝指南

> 📦 Make10 專案的詳細安裝說明與環境配置

## 🛠️ 系統需求

### 基本環境檢查
```powershell
# Python 版本檢查 (需要 3.12+)
python --version

# 系統資訊
systeminfo | findstr /B "OS Name"

# 可用記憶體 (建議 4GB+)
wmic computersystem get TotalPhysicalMemory
```

### PowerShell 權限設定
```powershell
# 檢查執行原則
Get-ExecutionPolicy

# 如果受限，設定權限
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## � 快速安裝 (推薦)

### 方法一：UV 套件管理器
```bash
# 安裝 UV (Windows PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 複製專案
git clone https://github.com/lingzinc/make10-solver.git
cd make10-solver

# 同步環境與依賴
uv sync --dev

# 驗證安裝
uv run pytest --version
```

## 🔧 詳細安裝步驟

### 步驟 1：環境準備
```bash
# 確認 Git 安裝
git --version

# 複製專案
git clone https://github.com/lingzinc/make10-solver.git
cd make10-solver

# 檢查專案結構
ls -la
```

### 步驟 2：UV 套件管理器安裝
```powershell
# 官方安裝指令
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 重新載入環境變數
refreshenv

# 驗證安裝
uv --version
```

### 步驟 3：Python 環境建立
```bash
# UV 會自動處理 Python 版本與虛擬環境
uv sync

# 檢查環境狀態
uv pip list

# 開發環境 (包含測試工具)
uv sync --dev
```

### 步驟 4：安裝驗證
```bash
# 執行測試確認環境
uv run pytest tests/ -v

# 檢查關鍵套件
uv run python -c "import cv2, numpy, tensorflow; print('環境正常')"

# 啟動系統測試
uv run run_system.py
```

## 🔄 替代安裝方法

### 方法二：傳統 pip 安裝
```bash
# 建立虛擬環境
python -m venv .venv

# 啟動環境 (Windows)
.venv\Scripts\activate

# 安裝依賴 (從 pyproject.toml)
pip install -e .[dev]

# 或直接安裝主要套件
pip install opencv-python==4.12.0.88 numpy==2.1.3 tensorflow==2.19.0 pynput==1.8.1 mss==10.0.0
```

### 方法三：Conda 環境
```bash
# 建立 Conda 環境
conda create -n make10 python=3.12
conda activate make10

# 安裝套件
conda install opencv numpy pandas
pip install tensorflow pynput mss loguru easydict

# 開發工具
pip install pytest pytest-cov ruff mypy
```

## 🧪 環境測試

### 功能驗證指令
```bash
# 完整測試套件
uv run pytest -v --cov=src

# 特定模組測試
uv run pytest tests/test_screen_utils.py -v

# 螢幕擷取測試
uv run python -c "from src.automation.screen_utils import capture_screen; print('螢幕擷取:', capture_screen() is not None)"

# 配置載入測試
uv run python -c "from config.settings import cfg; print('配置載入:', cfg.PROJECT_ROOT)"
```

### 系統相容性檢查
```bash
# OpenCV 功能測試
uv run python -c "import cv2; print('OpenCV版本:', cv2.__version__)"

# TensorFlow GPU 檢查
uv run python -c "import tensorflow as tf; print('GPU可用:', tf.config.list_physical_devices('GPU'))"

# 鍵盤監聽測試
uv run python -c "from src.automation.keyboard_listener import create_keyboard_listener; print('鍵盤監聽正常')"
```

## 🚨 常見問題排除

### OpenCV 安裝問題
```bash
# 重新安裝 OpenCV
uv pip uninstall opencv-python
uv pip install opencv-python==4.12.0.88

# 或使用 headless 版本
uv pip install opencv-python-headless
```

### TensorFlow 相容性
```bash
# 檢查 TensorFlow 支援
uv run python -c "import tensorflow as tf; print(tf.__version__)"

# CPU 版本 (如果 GPU 有問題)
uv pip install tensorflow-cpu==2.19.0
```

### 權限問題
```powershell
# 以管理員身分執行 PowerShell
Start-Process powershell -Verb runAs

# 或調整執行政策
Set-ExecutionPolicy Bypass -Scope Process -Force
```

### 套件衝突解決
```bash
# 清理環境重新安裝
uv pip freeze > requirements_backup.txt
uv pip uninstall -r requirements_backup.txt -y
uv sync --dev

# 檢查衝突
uv pip check
```

## 📊 效能最佳化

### Windows 系統最佳化
```powershell
# 關閉不必要的後台程式
Get-Process | Where-Object {$_.WorkingSet -gt 100MB} | Select-Object ProcessName, WorkingSet

# 檢查可用記憶體
Get-WmiObject -Class Win32_OperatingSystem | Select-Object @{Name="FreeMemory(GB)";Expression={[math]::round($_.FreePhysicalMemory/1MB,2)}}
```

### Python 環境最佳化
```bash
# 設定環境變數最佳化
set PYTHONOPTIMIZE=1
set TF_CPP_MIN_LOG_LEVEL=2

# 檢查套件安裝位置
uv pip show opencv-python tensorflow
```

## 🔗 相關資源

- 🐍 [Python 官方下載](https://python.org/downloads/)
- ⚡ [UV 官方文件](https://docs.astral.sh/uv/)
- 📷 [OpenCV 文件](https://docs.opencv.org/)
- 🧠 [TensorFlow 安裝指南](https://tensorflow.org/install)
- 🎮 [PyInput 文件](https://pynput.readthedocs.io/)
```bash
# 更新 pip
python -m pip install --upgrade pip

# 安裝 pipenv (替代虛擬環境管理)
pip install pipenv
```

## 🔧 專案設定

### 1. 專案複製
```bash
# 複製專案
git clone https://github.com/lingzinc/make10-solver.git

# 進入專案目錄
cd make10-solver

# 驗證專案結構
dir
# 應該看到: src/, tests/, config/, data/ 等目錄
```

### 2. 虛擬環境建立

#### 使用 UV (推薦)
```bash
# 建立虛擬環境
uv venv

# 啟動虛擬環境 (Windows)
.venv\Scripts\activate

# 安裝專案相依性
uv sync

# 安裝開發工具
uv add --dev pytest black flake8 mypy
```

#### 使用 Python venv (備用)
```bash
# 建立虛擬環境
python -m venv .venv

# 啟動虛擬環境
.venv\Scripts\activate

# 安裝相依性
pip install -r requirements.txt

# 安裝開發相依性
pip install -r requirements-dev.txt
```

### 3. 環境變數設定
```bash
# 設定 Python 路徑 (Windows PowerShell)
$env:PYTHONPATH = "."

# 持久性設定 (可選)
[Environment]::SetEnvironmentVariable("PYTHONPATH", ".", "User")
```

## 📋 相依性套件說明

### 核心相依性
```toml
# pyproject.toml 核心相依性
[dependencies]
tensorflow = ">=2.13.0"        # AI 模型框架
opencv-python = ">=4.8.0"      # 電腦視覺
numpy = ">=1.24.0"             # 數值計算
pillow = ">=10.0.0"            # 圖像處理
mss = ">=9.0.0"                # 螢幕擷取
pynput = ">=1.7.0"             # 輸入控制
pandas = ">=2.0.0"             # 資料處理
```

### 開發相依性
```toml
[dev-dependencies]
pytest = ">=7.0.0"             # 測試框架
black = ">=23.0.0"             # 程式碼格式化
flake8 = ">=6.0.0"             # 程式碼檢查
mypy = ">=1.0.0"               # 類型檢查
pytest-cov = ">=4.0.0"        # 覆蓋率測試
```

### 可選相依性
```bash
# 額外的機器學習套件
uv add scikit-learn matplotlib seaborn

# 效能分析工具
uv add memory-profiler line-profiler

# 文件產生工具
uv add sphinx sphinx-rtd-theme
```

## 🧪 安裝驗證

### 基本功能測試
```bash
# 1. 匯入測試
uv run python -c "
import tensorflow as tf
import cv2
import numpy as np
import PIL
from mss import mss
from pynput import mouse, keyboard
print('✅ 所有核心套件匯入成功')
"

# 2. 專案模組測試
uv run python -c "
from src.core import main
from src.automation import screen_utils
from src.automation import keyboard_listener
print('✅ 專案模組匯入成功')
"
```

### 系統整合測試
```bash
# 執行單元測試
uv run pytest tests/ -v

# 執行特定測試
uv run pytest tests/test_config_settings.py -v
uv run pytest tests/test_keyboard_listener.py -v
uv run pytest tests/test_screen_utils.py -v
```

### 效能測試
```bash
# TensorFlow GPU 支援檢查
uv run python -c "
import tensorflow as tf
print(f'TensorFlow 版本: {tf.__version__}')
print(f'GPU 可用: {tf.config.list_physical_devices(\"GPU\")}')
print(f'CPU 可用: {tf.config.list_physical_devices(\"CPU\")}')
"

# 記憶體使用測試
uv run python -c "
import psutil
print(f'可用記憶體: {psutil.virtual_memory().available / 1024**3:.2f} GB')
print(f'CPU 核心數: {psutil.cpu_count()}')
"
```

## 🔧 進階設定

### 開發工具設定

#### VS Code 設定
建立 `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": ".venv/Scripts/python.exe",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

#### Git Hooks 設定
```bash
# 安裝 pre-commit hooks
uv add --dev pre-commit
uv run pre-commit install

# 建立 .pre-commit-config.yaml
```

### 效能最佳化

#### TensorFlow 最佳化
```python
# 建立 config/tf_config.py
import tensorflow as tf

# GPU 記憶體增長設定
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# CPU 最佳化
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
```

#### OpenCV 最佳化
```python
# 建立 config/cv_config.py
import cv2

# 啟用多執行緒
cv2.setNumThreads(4)

# 設定 FFMPEG 後端
cv2.setUseOptimized(True)
```

## 📊 安裝後檢查清單

### 必要檢查項目
- [ ] Python 3.12+ 正確安裝
- [ ] UV 套件管理器可用
- [ ] 虛擬環境建立成功
- [ ] 所有相依性套件安裝完成
- [ ] 單元測試執行通過
- [ ] 專案模組可正常匯入

### 功能檢查項目
- [ ] TensorFlow 可正常執行
- [ ] OpenCV 圖像處理功能正常
- [ ] 螢幕擷取功能可用
- [ ] 鍵盤滑鼠控制功能正常
- [ ] 設定檔載入無錯誤

### 整合檢查項目
- [ ] 系統主程式可執行
- [ ] AI 模型載入成功
- [ ] 遊戲畫面偵測正常
- [ ] 自動化流程運作順暢

## 🛠️ 故障排除

### 常見安裝問題

#### 問題 1: UV 安裝失敗
```bash
# 解決方案 1: 檢查網路連線
ping github.com

# 解決方案 2: 手動下載安裝
# 訪問 https://github.com/astral-sh/uv/releases
# 下載對應平台版本手動安裝
```

#### 問題 2: TensorFlow 安裝問題
```bash
# 解決方案: 指定版本安裝
uv add "tensorflow==2.13.0"

# CPU 版本 (如果 GPU 有問題)
uv add tensorflow-cpu
```

#### 問題 3: OpenCV 匯入錯誤
```bash
# 解決方案: 重新安裝 OpenCV
uv remove opencv-python
uv add opencv-python-headless
```

#### 問題 4: 權限問題
```powershell
# 解決方案: 以管理員身分執行
Start-Process PowerShell -Verb RunAs
```

### 環境相容性問題

#### Python 版本相容性
```bash
# 檢查支援的 Python 版本
uv run python -c "
import sys
print(f'Python 版本: {sys.version}')
if sys.version_info >= (3, 12):
    print('✅ Python 版本相容')
else:
    print('❌ 需要 Python 3.12+')
"
```

#### 套件版本衝突
```bash
# 檢查套件相依性
uv show

# 解決版本衝突
uv sync --resolution=highest
```

## 📈 安裝效能監控

### 安裝時間基準
- **基本安裝**: 5-10 分鐘
- **完整安裝** (含開發工具): 10-15 分鐘
- **驗證測試**: 2-5 分鐘

### 資源使用監控
```bash
# 安裝過程中監控資源使用
uv run python -c "
import psutil
import time

for i in range(60):
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    print(f'CPU: {cpu}%, 記憶體: {mem}%')
    time.sleep(1)
"
```

安裝完成後，您可以繼續參考 [`getting-started.md`](./getting-started.md) 進行系統的快速設定與測試。
