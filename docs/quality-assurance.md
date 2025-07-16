# 程式碼品質保證指南

✨ 本指南說明 Make10 專案的程式碼品質標準、CI/CD 流程與自動化檢查機制。

## 🎯 程式碼品質標準

### 品質目標
- **測試覆蓋率**: > 80%
- **程式碼重複率**: < 5%
- **圈複雜度**: < 10 (每個函式)
- **文件完整度**: > 90%
- **靜態分析**: 0 個嚴重問題

### 程式碼風格規範
遵循 PEP 8 標準，並使用自動化工具確保一致性。

## 🛠️ 自動化工具配置

### Black - 程式碼格式化
```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py312']
include = '\.pyi?$'
extend-exclude = '''
/(
    __pycache__
    | \.git
    | \.venv
    | build
    | dist
    | \.mypy_cache
    | \.pytest_cache
)/
'''
```

#### 使用方式
```bash
# 格式化所有檔案
uv run black .

# 檢查格式但不修改
uv run black --check .

# 格式化特定檔案
uv run black src/core/main.py

# 顯示會做的變更
uv run black --diff .
```

### Flake8 - 程式碼檢查
```ini
# .flake8
[flake8]
max-line-length = 88
max-complexity = 10
ignore = 
    E203,  # 空白符號 (與 Black 衝突)
    W503,  # 二元運算符前的換行 (與 Black 衝突)
    E501   # 行長度 (由 Black 處理)
exclude = 
    .git,
    __pycache__,
    .venv,
    build,
    dist,
    *.egg-info,
    .mypy_cache,
    .pytest_cache
per-file-ignores =
    __init__.py:F401  # 允許 __init__.py 中的未使用匯入
    tests/*:S101      # 允許測試中使用 assert
```

#### 使用方式
```bash
# 檢查所有檔案
uv run flake8

# 檢查特定目錄
uv run flake8 src/

# 檢查特定檔案
uv run flake8 src/core/main.py

# 顯示統計資訊
uv run flake8 --statistics
```

### MyPy - 類型檢查
```toml
# pyproject.toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = [
    "cv2",
    "mss",
    "pynput",
    "PIL",
]
ignore_missing_imports = true
```

#### 使用方式
```bash
# 檢查所有檔案
uv run mypy src/

# 檢查特定檔案
uv run mypy src/core/main.py

# 生成詳細報告
uv run mypy --html-report mypy-report src/
```

### 類型註解範例
```python
from typing import List, Optional, Dict, Tuple, Union
import numpy as np
from pathlib import Path

class ModelManager:
    """AI 模型管理器"""
    
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._model: Optional[tf.keras.Model] = None
    
    def load_model(self) -> bool:
        """載入模型
        
        Returns:
            bool: 載入成功則返回 True
        """
        try:
            self._model = tf.keras.models.load_model(str(self.model_path))
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_batch(
        self, 
        images: List[np.ndarray]
    ) -> Tuple[List[int], List[float]]:
        """批次預測
        
        Args:
            images: 輸入圖像列表
            
        Returns:
            預測結果和信心度的元組
            
        Raises:
            ValueError: 當模型未載入時
        """
        if self._model is None:
            raise ValueError("Model not loaded")
        
        # 實作預測邏輯
        predictions: List[int] = []
        confidences: List[float] = []
        
        return predictions, confidences
```

## 🧪 測試品質保證

### pytest 配置
```toml
# pyproject.toml
[tool.pytest.ini_options]
minversion = "6.0"
addopts = [
    "-ra",
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=term-missing:skip-covered",
    "--cov-report=html:htmlcov",
    "--cov-report=xml",
    "--cov-fail-under=80",
]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "unit: 單元測試",
    "integration: 整合測試",
    "slow: 慢速測試",
    "ai: AI 相關測試",
    "automation: 自動化測試",
]
```

### 測試覆蓋率配置
```toml
# pyproject.toml
[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*",
    "*/conftest.py",
    "*/setup.py",
]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
show_missing = true
skip_covered = false
```

### 測試品質檢查
```bash
# 執行所有測試並生成覆蓋率報告
uv run pytest --cov=src --cov-report=html

# 檢查測試覆蓋率是否達標
uv run pytest --cov=src --cov-fail-under=80

# 執行特定標記的測試
uv run pytest -m "unit and not slow"

# 檢查測試效能
uv run pytest --durations=10
```

## 🔍 靜態分析工具

### Bandit - 安全性檢查
```bash
# 安裝 Bandit
uv add --dev bandit

# 檢查安全性問題
uv run bandit -r src/

# 生成詳細報告
uv run bandit -r src/ -f json -o bandit-report.json
```

### Pylint - 深度程式碼分析
```toml
# pyproject.toml
[tool.pylint.messages_control]
disable = [
    "missing-docstring",
    "too-few-public-methods",
    "too-many-arguments",
    "too-many-locals",
    "duplicate-code",
]

[tool.pylint.format]
max-line-length = 88

[tool.pylint.design]
max-complexity = 10
```

```bash
# 安裝並執行 Pylint
uv add --dev pylint
uv run pylint src/
```

### Pre-commit Hooks 設定
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements
  
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.12
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
  
  - repo: https://github.com/pycqa/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-c', 'pyproject.toml']
  
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: uv run pytest
        language: system
        pass_filenames: false
        always_run: true
        stages: [commit]
```

#### 安裝和使用
```bash
# 安裝 pre-commit
uv add --dev pre-commit

# 安裝 hooks
uv run pre-commit install

# 手動執行所有 hooks
uv run pre-commit run --all-files

# 更新 hooks
uv run pre-commit autoupdate
```

## 🚀 CI/CD 管道

### GitHub Actions 工作流程
```yaml
# .github/workflows/quality-check.yml
name: Code Quality Check

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  quality-check:
    runs-on: windows-latest
    strategy:
      matrix:
        python-version: [3.12]

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install UV
      run: |
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

    - name: Install dependencies
      run: |
        uv sync

    - name: Format check with Black
      run: |
        uv run black --check .

    - name: Lint with Flake8
      run: |
        uv run flake8

    - name: Type check with MyPy
      run: |
        uv run mypy src/

    - name: Security check with Bandit
      run: |
        uv run bandit -r src/

    - name: Run tests with coverage
      run: |
        uv run pytest --cov=src --cov-report=xml --cov-report=term

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

    - name: Generate quality report
      if: always()
      run: |
        echo "## Code Quality Report" >> $GITHUB_STEP_SUMMARY
        echo "### Test Coverage" >> $GITHUB_STEP_SUMMARY
        uv run coverage report --format=markdown >> $GITHUB_STEP_SUMMARY || true
```

### 品質門檻設定
```yaml
# .github/workflows/quality-gate.yml
name: Quality Gate

on:
  pull_request:
    branches: [ main ]

jobs:
  quality-gate:
    runs-on: windows-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0  # 需要完整歷史來比較

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.12

    - name: Install dependencies
      run: |
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
        uv sync

    - name: Check code quality metrics
      run: |
        # 測試覆蓋率檢查
        uv run pytest --cov=src --cov-fail-under=80
        
        # 複雜度檢查
        uv run radon cc src/ --min B
        
        # 重複程式碼檢查
        uv run pylint src/ --disable=all --enable=duplicate-code

    - name: Performance regression test
      run: |
        uv run pytest tests/test_performance.py --benchmark-only

    - name: Security scan
      run: |
        uv run bandit -r src/ --severity-level medium

    - name: Quality gate summary
      if: always()
      run: |
        echo "Quality gate checks completed"
        echo "✅ Code coverage: $(uv run coverage report --precision=1 | grep TOTAL | awk '{print $4}')"
        echo "✅ Security issues: $(uv run bandit -r src/ -f json | jq '.results | length')"
        echo "✅ Type coverage: $(uv run mypy src/ --txt-report mypy-report | grep -o '[0-9]*%' | tail -1)"
```

## 📊 程式碼品質監控

### SonarQube 整合
```yaml
# sonar-project.properties
sonar.projectKey=make10-solver
sonar.projectName=Make10 Solver
sonar.projectVersion=1.0.0
sonar.sources=src
sonar.tests=tests
sonar.python.coverage.reportPaths=coverage.xml
sonar.python.xunit.reportPath=pytest-report.xml
sonar.exclusions=**/__pycache__/**,**/venv/**,**/build/**
```

### 品質指標儀表板
```python
# scripts/quality_dashboard.py
import json
import subprocess
from pathlib import Path

def generate_quality_report():
    """生成程式碼品質報告"""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "metrics": {}
    }
    
    # 測試覆蓋率
    coverage_result = subprocess.run(
        ["uv", "run", "coverage", "report", "--format=json"],
        capture_output=True, text=True
    )
    if coverage_result.returncode == 0:
        coverage_data = json.loads(coverage_result.stdout)
        report["metrics"]["coverage"] = coverage_data["totals"]["percent_covered"]
    
    # 複雜度分析
    complexity_result = subprocess.run(
        ["uv", "run", "radon", "cc", "src/", "--json"],
        capture_output=True, text=True
    )
    if complexity_result.returncode == 0:
        complexity_data = json.loads(complexity_result.stdout)
        avg_complexity = calculate_average_complexity(complexity_data)
        report["metrics"]["complexity"] = avg_complexity
    
    # 安全性問題
    security_result = subprocess.run(
        ["uv", "run", "bandit", "-r", "src/", "-f", "json"],
        capture_output=True, text=True
    )
    if security_result.returncode == 0:
        security_data = json.loads(security_result.stdout)
        report["metrics"]["security_issues"] = len(security_data["results"])
    
    # 技術債務
    debt_result = subprocess.run(
        ["uv", "run", "pylint", "src/", "--output-format=json"],
        capture_output=True, text=True
    )
    if debt_result.returncode == 0:
        debt_data = json.loads(debt_result.stdout)
        report["metrics"]["technical_debt"] = calculate_debt_score(debt_data)
    
    # 儲存報告
    report_path = Path("quality-report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report

def calculate_average_complexity(complexity_data):
    """計算平均複雜度"""
    total_complexity = 0
    total_functions = 0
    
    for module_data in complexity_data.values():
        for class_or_func in module_data:
            if class_or_func["type"] in ["function", "method"]:
                total_complexity += class_or_func["complexity"]
                total_functions += 1
    
    return total_complexity / total_functions if total_functions > 0 else 0

if __name__ == "__main__":
    report = generate_quality_report()
    print(f"Quality report generated: {json.dumps(report, indent=2)}")
```

## 🎯 最佳實務建議

### 程式碼審查檢查表
#### 功能性
- [ ] 程式碼實現符合需求
- [ ] 邊界條件處理正確
- [ ] 錯誤處理機制完善
- [ ] 效能符合預期

#### 可讀性
- [ ] 變數和函式命名清晰
- [ ] 程式碼結構邏輯清楚
- [ ] 註解說明充分
- [ ] 複雜邏輯有文件說明

#### 可維護性
- [ ] 函式長度適中 (< 50 行)
- [ ] 重複程式碼已抽取
- [ ] 依賴關係清晰
- [ ] 配置與程式碼分離

#### 測試
- [ ] 單元測試覆蓋率足夠
- [ ] 測試案例有代表性
- [ ] 測試容易理解和維護
- [ ] 整合測試涵蓋主要流程

### 持續改進流程
1. **每週品質回顧**: 檢視品質指標趨勢
2. **技術債務清理**: 定期重構低品質程式碼
3. **工具升級**: 定期更新分析工具版本
4. **團隊培訓**: 分享最佳實務和新工具

### 品質文化建立
- **責任共擔**: 每個開發者都對程式碼品質負責
- **持續學習**: 定期學習新的品質保證方法
- **工具自動化**: 最大化自動化檢查，減少人工錯誤
- **快速反饋**: 在開發早期發現和修復問題

透過這套完整的程式碼品質保證體系，Make10 專案能夠維持高水準的程式碼品質，確保系統的穩定性和可維護性。
