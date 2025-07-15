## 開發環境說明

- 作業系統：Windows 11
- 使用 Python 3.12，搭配 `uv` 來管理虛擬環境（非 `venv` 或 `pip`）。
- 所有 Python 套件統一安裝在 `.venv`，建立方式如下：

```bash
uv venv .venv
```

## 使用方法

### 執行主程式

使用 `uv run` 執行專案，這會自動管理虛擬環境和相依性：

```bash
# 執行系統主程式（推薦）
uv run run_system.py

# 執行AI模型訓練
uv run run_training.py

# 或直接執行核心模組
uv run src/core/main.py
```

### 安裝相依套件

```bash
# 同步安裝所有相依套件
uv sync

# 安裝開發相依套件
uv sync --dev
```

### 其他常用指令

```bash
# 檢查程式碼品質
uv run ruff check src/

# 格式化程式碼
uv run ruff format src/

# 執行測試
uv run pytest

# 類型檢查
uv run mypy src/
```