# 快速入門指南

> 🚀 Make10 遊戲自動化系統的快速上手指南

## 📋 系統需求

### 基本環境
- **作業系統**: Windows 10/11 
- **Python**: 3.12+ (必要)
- **記憶體**: 4GB+ RAM (建議 8GB+)
- **硬碟**: 2GB 可用空間

### 必要工具
- **UV**: 現代化 Python 套件管理器 (推薦)
- **Git**: 版本控制 (必要)

## ⚡ 一分鐘快速安裝

### 方法一：UV 套件管理器 (推薦)
```bash
# 1. 安裝 UV
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 複製並設定專案
git clone https://github.com/lingzinc/make10-solver.git
cd make10-solver
uv sync --dev

# 3. 驗證安裝
uv run pytest tests/ -x
```

### 方法二：傳統 pip 方式
```bash
# 1. 複製專案
git clone https://github.com/lingzinc/make10-solver.git
cd make10-solver

# 2. 建立虛擬環境
python -m venv .venv
.venv\Scripts\activate

# 3. 安裝套件
pip install -e .[dev]
```

## 🎮 立即體驗

### 啟動系統
```bash
# 啟動 Make10 自動化系統
uv run run_system.py

# 預期輸出:
# === Make10 遊戲自動化系統 v0.1.0 ===
# [INFO] 系統啟動中...
# [INFO] 相依套件檢查通過
# [INFO] 系統核心模組初始化完成
```

### 熱鍵操作
- **Ctrl+Q** - 安全退出系統
- **Alt+Tab** - 自動螢幕切換

### 基本功能測試
```bash
# 測試螢幕擷取功能
uv run python -c "
from src.automation.screen_utils import capture_screen
screenshot = capture_screen()
print(f'螢幕擷取: {\"成功\" if screenshot is not None else \"失敗\"}')
"

# 測試配置系統
uv run python -c "
from config.settings import cfg
print(f'配置載入: 成功')
print(f'模型路徑: {cfg.PATHS.MODEL.main_model}')
"
```

## 📊 系統架構概覽

### 目前可用功能 ✅
```
系統啟動 ──┬── 日誌系統      (完整功能)
           ├── 配置管理      (完整功能)
           ├── 螢幕擷取      (完整功能)
           ├── 鍵盤監聽      (完整功能)
           └── 自動化控制    (基礎功能)
```

### 開發中功能 🚧
```
AI 智慧識別 ──┬── 數字識別     (規劃中)
              ├── 模型訓練     (開發中)
              └── 遊戲求解     (規劃中)
```

## 🧪 驗證安裝成功

### 執行測試套件
```bash
# 完整測試
uv run pytest -v

# 特定模組測試
uv run pytest tests/test_config_settings.py -v
uv run pytest tests/test_screen_utils.py -v
uv run pytest tests/test_keyboard_listener.py -v

# 覆蓋率測試
uv run pytest --cov=src --cov-report=term-missing
```

### 檢查關鍵元件
```bash
# 核心模組
uv run python -c "from src.core.main import GameAutomationSystem; print('✅ 核心系統')"

# 自動化模組
uv run python -c "from src.automation import screen_utils, keyboard_listener; print('✅ 自動化模組')"

# 配置系統
uv run python -c "from config.settings import cfg; print('✅ 配置系統')"
```

## 🎯 基本使用流程

### 1. 啟動系統
```bash
# 在專案根目錄執行
cd make10-solver
uv run run_system.py
```

### 2. 監控日誌
```bash
# 另開終端視窗監控日誌
tail -f logs/make10_system.log

# 或使用 PowerShell
Get-Content logs/make10_system.log -Wait
```

### 3. 安全退出
- 按下 **Ctrl+Q** 組合鍵
- 或在終端按 **Ctrl+C**

## 🔧 常見問題快速修復

### UV 安裝問題
```powershell
# 如果 UV 安裝失敗，檢查執行政策
Get-ExecutionPolicy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 重新安裝
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 套件安裝問題
```bash
# 清理並重新安裝
uv clean
uv sync --dev

# 或使用 pip 替代
pip install -e .[dev]
```

### 權限問題
```powershell
# 以管理員身分執行 PowerShell
Start-Process powershell -Verb runAs
cd make10-solver
uv run run_system.py
```

## 📚 下一步學習

### 開發者路徑
1. 📖 [開發指南](./development-guide.md) - 了解開發流程
2. 🏗️ [專案架構](./project-structure.md) - 深入專案結構  
3. 🧪 [測試指南](./testing-guide.md) - 學習測試方法

### 技術深入
1. 🔧 [技術架構](./technical-architecture.md) - 系統架構設計
2. 🧠 [AI 模型指南](./ai-model-guide.md) - AI 模型開發
3. ⚙️ [安裝指南](./installation.md) - 詳細安裝說明

### 問題解決
1. 🚨 [故障排除](./troubleshooting.md) - 常見問題解決
2. 📊 [系統工作流程](./system-workflow.md) - 了解系統運作

## 🤝 參與專案

### 提出問題
- 🐙 [GitHub Issues](https://github.com/lingzinc/make10-solver/issues)
- 💬 提交 Bug 回報或功能請求

### 貢獻程式碼
```bash
# Fork 專案並建立功能分支
git checkout -b feature/your-feature-name

# 開發並測試
uv run pytest tests/ -v

# 提交 Pull Request
```

## 💡 小貼士

### 效能最佳化
- 確保有足夠的 RAM (建議 8GB+)
- 關閉不必要的後台程式
- 定期清理日誌檔案

### 開發環境建議
- 使用 **VS Code** 搭配 Python 外掛
- 安裝 **Ruff** 程式碼格式化工具
- 啟用 **MyPy** 型別檢查

### 最佳實務
- 定期執行測試確保程式品質
- 使用 Git 分支管理功能開發
- 遵循專案的程式碼規範

---

🎉 **恭喜！** 您已經成功設定 Make10 自動化系統。開始探索更多進階功能吧！

## 🎮 第一次執行

### 系統模式執行
```bash
# 執行完整自動化系統
uv run python run_system.py

# 或使用核心模組
uv run python src/core/main.py
```

### 訓練模式執行
```bash
# 訓練 AI 模型
uv run python run_training.py

# 使用標註模式
uv run python run_training.py --label
```

## 🎯 基本使用流程

### 1. 準備遊戲環境
- 開啟 Make10 遊戲
- 確保遊戲視窗可見且未被遮擋
- 建議將遊戲置於主螢幕

### 2. 執行自動化系統
```bash
uv run python run_system.py
```

### 3. 系統操作
- 系統會自動偵測遊戲畫面
- 執行盤面掃描與數字識別
- 計算最佳解答
- 自動執行遊戲操作

## ⚡ 快速驗證檢查表

### 環境檢查
- [ ] Python 3.12+ 已安裝
- [ ] UV 套件管理器可用
- [ ] 虛擬環境已建立並啟動
- [ ] 專案相依性已安裝

### 系統功能檢查
- [ ] 螢幕擷取功能正常
- [ ] 鍵盤監聽器可運作
- [ ] AI 模型可正常載入
- [ ] 測試套件執行通過

### 遊戲整合檢查
- [ ] 遊戲視窗可被偵測
- [ ] 重置按鈕識別正常
- [ ] 數字識別準確度可接受
- [ ] 自動化操作執行順暢

## 🔧 常見問題快速解決

### 安裝問題
**Q: UV 安裝失敗**
```bash
# 解決方案：使用 pip 作為替代方案
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Q: 虛擬環境無法啟動**
```bash
# 解決方案：手動建立虛擬環境
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 執行問題
**Q: 模組匯入錯誤**
```bash
# 解決方案：設定 Python 路徑
$env:PYTHONPATH = "."
uv run python run_system.py
```

**Q: AI 模型載入失敗**
```bash
# 解決方案：重新訓練模型
uv run python run_training.py --new
```

## 📚 下一步

設定完成後，建議您：

1. **閱讀系統工作流程**: [`system-workflow.md`](./system-workflow.md)
2. **了解專案架構**: [`project-structure.md`](./project-structure.md)
3. **深入開發指南**: [`development-guide.md`](./development-guide.md)

## 🆘 需要協助？

如果遇到問題，請參考：
- [`troubleshooting.md`](./troubleshooting.md) - 詳細故障排除指南
- [`installation.md`](./installation.md) - 詳細安裝說明
