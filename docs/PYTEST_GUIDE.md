# pytest 測試執行指南

## 📋 **基本執行命令**

### 1. 執行所有測試
```bash
uv run pytest
```

### 2. 執行特定測試檔案
```bash
uv run pytest tests/test_config_settings.py
```

### 3. 執行特定測試類別
```bash
uv run pytest tests/test_config_settings.py::TestConfigurationObject
```

### 4. 執行特定測試方法
```bash
uv run pytest tests/test_config_settings.py::TestConfigurationValues::test_model_values
```

## 🔍 **測試輸出選項**

### 詳細輸出 (-v)
```bash
uv run pytest -v
```

### 簡潔輸出 (-q)
```bash
uv run pytest -q
```

### 顯示本地變數 (-l)
```bash
uv run pytest -l
```

### 詳細錯誤追蹤
```bash
uv run pytest --tb=long    # 完整追蹤
uv run pytest --tb=short   # 簡短追蹤
uv run pytest --tb=line    # 單行追蹤
uv run pytest --tb=no      # 不顯示追蹤
```

## 🏷️ **使用標記**

### 執行特定標記的測試
```bash
uv run pytest -m unit          # 只執行單元測試
uv run pytest -m integration   # 只執行整合測試
uv run pytest -m "not slow"    # 排除慢速測試
```

## 🔄 **測試重複執行**

### 重複執行失敗的測試
```bash
uv run pytest --lf   # last-failed
```

### 重複執行最後失敗的測試並停止在第一個失敗
```bash
uv run pytest --ff   # failed-first
```

### 執行指定次數
```bash
uv run pytest --count=3
```

## 🚫 **停止條件**

### 第一個失敗時停止
```bash
uv run pytest -x
```

### 指定失敗次數後停止
```bash
uv run pytest --maxfail=2
```

## 📊 **測試報告**

### 覆蓋率報告
```bash
uv run pytest --cov=. --cov-report=term-missing   # 終端顯示缺失行數
```

### JUnit XML 報告
```bash
uv run pytest --junitxml=reports/junit.xml
```

### HTML 報告（需要 pytest-html）
```bash
uv run pytest --html=reports/report.html
```

## 🧪 **我們的 config 測試**

### 執行配置物件結構測試
```bash
uv run pytest tests/test_config_settings.py::TestConfigurationObject -v
```

### 執行路徑驗證測試
```bash
uv run pytest tests/test_config_settings.py::TestPathValidation -v
```

### 執行配置值測試
```bash
uv run pytest tests/test_config_settings.py::TestConfigurationValues -v
```

### 執行模型路徑測試
```bash
uv run pytest tests/test_config_settings.py::TestModelPath -v
```

### 執行除錯目錄測試
```bash
uv run pytest tests/test_config_settings.py::TestDebugDirectory -v
```

### 執行初始化設定測試
```bash
uv run pytest tests/test_config_settings.py::TestInitializeSettings -v
```

## 📈 **測試結果範例**

成功的測試輸出：
```
======================== test session starts ========================
platform win32 -- Python 3.12.11, pytest-8.4.1
collected 22 items

tests/test_config_settings.py ......................        [100%]

======================== 22 passed in 0.34s ========================
```

## 🛠️ **除錯技巧**

### 顯示 print 輸出
```bash
uv run pytest -s
```

### 進入除錯模式
```bash
uv run pytest --pdb
```

### 在第一個失敗時進入除錯
```bash
uv run pytest --pdb-trace
```

## 📝 **常用組合**

### 開發時的完整測試
```bash
uv run pytest -v --tb=short
```

### 快速驗證
```bash
uv run pytest -q
```

### 詳細除錯
```bash
uv run pytest -v -s --tb=long
```
