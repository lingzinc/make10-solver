# Make10 開發者文件中心

> 📚 Make10 自動化系統技術文件 - 為開發者提供快速參閱的技術指南

## �️ 文件導航

### 🚀 入門指南
| 文件 | 描述 | 適用對象 |
|------|------|----------|
| [`getting-started.md`](./getting-started.md) | 環境設定與基本概念 | 新手開發者 |
| [`installation.md`](./installation.md) | 詳細安裝與依賴配置 | 系統管理員 |

### 🏗️ 架構文件
| 文件 | 描述 | 適用對象 |
|------|------|----------|
| [`technical-architecture.md`](./technical-architecture.md) | 系統架構與核心演算法 | 架構師/資深開發者 |
| [`project-structure.md`](./project-structure.md) | 專案結構與模組關係 | 所有開發者 |
| [`development-guide.md`](./development-guide.md) | 開發流程與規範 | 開發團隊 |

### 🧠 AI/ML 文件
| 文件 | 描述 | 適用對象 |
|------|------|----------|
| [`ai-model-guide.md`](./ai-model-guide.md) | AI 模型架構與訓練 | ML 工程師 |
| [`training-workflow.md`](./training-workflow.md) | 模型訓練流程 | 資料科學家 |

### 🧪 品質保證
| 文件 | 描述 | 適用對象 |
|------|------|----------|
| [`testing-guide.md`](./testing-guide.md) | 測試策略與執行 | QA/開發者 |
| [`quality-assurance.md`](./quality-assurance.md) | 程式碼品質與 CI/CD | DevOps 工程師 |

### � 操作指南
| 文件 | 描述 | 適用對象 |
|------|------|----------|
| [`system-workflow.md`](./system-workflow.md) | 系統執行流程 | 操作員/測試員 |
| [`troubleshooting.md`](./troubleshooting.md) | 故障排除手冊 | 技術支援 |

## 🎯 快速索引

### 常用程式碼範例
```python
# 系統啟動
from src.core.main import main
main()

# 螢幕擷取
from src.automation.screen_utils import capture_screen
screenshot = capture_screen()

# 配置存取
from config.settings import cfg
model_path = cfg.PATHS.MODEL.main_model
```

### 關鍵配置參數
```python
# 自動化參數
cfg.AUTOMATION.click_delay      # 點擊延遲
cfg.AUTOMATION.retry_attempts   # 重試次數

# AI 模型參數  
cfg.MODEL.confidence_threshold  # 信心閾值
cfg.MODEL.batch_size           # 批次大小

# 影像處理參數
cfg.IMAGE.PROCESSING.cell_size  # 格子大小
cfg.IMAGE.PROCESSING.board_size # 棋盤大小
```

### 測試指令
```bash
# 完整測試套件
uv run pytest -v --cov=src

# 特定模組測試
uv run pytest tests/test_screen_utils.py -v

# 覆蓋率報告
uv run pytest --cov=src --cov-report=html
```

## 📋 文件撰寫規範

### 格式標準
- 使用 Markdown 格式
- 程式碼區塊標注語言類型
- 包含實用範例與程式碼片段
- 保持簡潔，避免冗長說明

### 更新原則
- 程式碼變更時同步更新文件
- 新增功能必須包含文件
- 定期檢查文件的準確性
- 移除過時或無用的內容

## 🔗 相關資源

- 🏠 [專案首頁](../README.md)
- 📦 [PyPI 套件管理](https://pypi.org/)
- 🛠️ [UV 文件](https://docs.astral.sh/uv/)
- 🧪 [Pytest 測試框架](https://pytest.org/)
- 🎨 [Ruff 程式碼格式化](https://github.com/astral-sh/ruff)

- **建立日期**: 2025年7月16日
- **最後更新**: 2025年7月16日
- **維護者**: make10 開發團隊
- **版本控制**: 所有文件變更均透過 Git 進行版本控制

## 🔄 文件更新原則

1. **內容準確性**: 確保所有技術資訊與程式碼實作保持同步
2. **結構清晰**: 按功能與使用情境分類，避免重複內容
3. **易於維護**: 單一責任原則，每個文件專注於特定主題
4. **持續改進**: 根據使用者反饋持續優化文件品質
