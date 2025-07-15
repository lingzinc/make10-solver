# Make10 遊戲自動化系統

🎯 基於 AI 和電腦視覺技術的自動化遊戲求解器

## 🚀 快速開始

### 📋 系統需求

#### 基本環境
- **作業系統**: Windows 10/11
- **Python**: 3.12+ (建議使用最新版本)

#### 必要工具
- **套件管理器**: [UV](https://docs.astral.sh/uv/) (現代化 Python 套件管理工具)
- **Git**: 用於版本控制和專案複製

### 🔧 環境設定

#### 步驟一: 安裝 UV 套件管理器
```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 步驟二: 複製專案
```bash
git clone https://github.com/lingzinc/make10-solver.git
cd make10-solver
```

#### 步驟三: 建立虛擬環境並安裝相依性
```bash
# 建立虛擬環境
uv venv .venv

# 安裝所有相依性（包含開發工具）
uv sync --dev

# 啟動虛擬環境（可選）
.venv\Scripts\activate
```

### 🎯 專案執行

#### 🎮 執行自動化系統
```bash
# 啟動 Make10 遊戲自動化求解器
uv run run_system.py
```

#### 🧠 執行 AI 模型訓練
```bash
# 訓練數字識別模型
uv run run_training.py
```

### 🧪 品質保證

```bash
# 執行測試
uv run pytest -v                                # 執行所有測試
uv run pytest --cov=. --cov-report=term-missing # 顯示覆蓋率和缺失行數
```

## ⚡ 技術架構

```
Make10 自動化系統
├── 🎯 遊戲檢測 (OpenCV + 霍夫變換)
├── 🧠 數字識別 (TensorFlow CNN)
├── 🎮 自動操作 (PyInput + MSS)
└── 📊 數據分析 (Pandas + NumPy)
```
