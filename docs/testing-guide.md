# 測試指南

> 🧪 Make10 專案的測試策略、執行方法與品質保證

## 🎯 測試策略

### 測試金字塔
```
    🔺 端到端測試 (5%)
   🔸🔸🔸 整合測試 (20%)  
 🔹🔹🔹🔹🔹 單元測試 (75%)
```

### 測試分類與目標
- **單元測試** - 測試個別函式與類別 (目標 >90% 覆蓋率)
- **整合測試** - 測試模組間互動 (重點測試 API 介面)
- **端到端測試** - 測試完整使用者流程 (關鍵功能覆蓋)
- **效能測試** - 驗證系統效能指標 (響應時間 <100ms)

## 🚀 測試執行

### 基本指令
```bash
# 執行所有測試
uv run pytest -v

# 特定檔案測試
uv run pytest tests/test_screen_utils.py -v

# 特定類別測試
uv run pytest tests/test_screen_utils.py::TestScreenUtils -v

# 特定測試方法
uv run pytest tests/test_config_settings.py::TestConfigurationObject::test_paths_model -v
```

### 覆蓋率測試
```bash
# 基本覆蓋率報告
uv run pytest --cov=src

# 詳細覆蓋率 (顯示缺失行數)
uv run pytest --cov=src --cov-report=term-missing

# HTML 覆蓋率報告
uv run pytest --cov=src --cov-report=html

# 覆蓋率門檻驗證
uv run pytest --cov=src --cov-fail-under=80
```

### 進階選項
```bash
# 快速失敗 (第一個失敗即停止)
uv run pytest -x

# 最大失敗數限制
uv run pytest --maxfail=3

# 只重新執行失敗的測試
uv run pytest --lf

# 顯示執行最慢的測試
uv run pytest --durations=10
```

## 📋 測試分類與標記

### 測試標記系統
```python
# pytest.ini 配置
[pytest]
markers =
    unit: 單元測試
    integration: 整合測試
    slow: 需要較長時間的測試
    ai: AI 相關測試
    automation: 自動化功能測試
    config: 配置相關測試
    visual: 視覺相關測試
```

### 使用標記執行測試
```bash
# 只執行單元測試
uv run pytest -m unit

# 只執行整合測試
uv run pytest -m integration

# 排除慢速測試
uv run pytest -m "not slow"

# 執行 AI 相關測試
uv run pytest -m ai

# 組合標記
uv run pytest -m "unit and not slow"
```

## 🧪 測試類型詳解

### 1. 單元測試 (Unit Tests)

#### 配置模組測試範例
```python
# tests/test_config_settings.py
import pytest
from pathlib import Path
from config.settings import Settings, initialize_settings

class TestConfigurationObject:
    """配置物件結構測試"""
    
    def test_settings_has_required_attributes(self):
        """測試設定物件包含必要屬性"""
        settings = Settings()
        
        required_attrs = [
            'model_path', 'data_path', 'log_path',
            'debug_mode', 'move_delay', 'max_retries'
        ]
        
        for attr in required_attrs:
            assert hasattr(settings, attr), f"Settings 缺少必要屬性: {attr}"
    
    def test_settings_types(self):
        """測試設定屬性類型"""
        settings = Settings()
        
        assert isinstance(settings.model_path, Path)
        assert isinstance(settings.move_delay, (int, float))
        assert isinstance(settings.debug_mode, bool)
        assert isinstance(settings.max_retries, int)

class TestPathValidation:
    """路徑驗證測試"""
    
    def test_model_path_exists(self):
        """測試模型路徑存在性"""
        settings = Settings()
        
        # 如果模型檔案存在，路徑應該是有效的
        if settings.model_path.exists():
            assert settings.model_path.is_file()
            assert settings.model_path.suffix == '.keras'
    
    def test_data_directory_structure(self):
        """測試資料目錄結構"""
        settings = Settings()
        
        # 檢查必要的子目錄
        required_subdirs = ['models', 'training', 'assets']
        for subdir in required_subdirs:
            subdir_path = settings.data_path / subdir
            assert subdir_path.exists(), f"資料子目錄不存在: {subdir}"
```

#### AI 模組測試範例
```python
# tests/test_ai_predictor.py
import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.ai.predictor import Predictor
from src.ai.image_processor import ImageProcessor

class TestPredictor:
    """AI 預測器測試"""
    
    @pytest.fixture
    def mock_model(self):
        """模擬 TensorFlow 模型"""
        model = Mock()
        model.predict.return_value = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 
                                              0.1, 0.1, 0.1, 0.1, 0.1]])
        return model
    
    @pytest.fixture
    def predictor(self, mock_model):
        """建立預測器實例"""
        return Predictor(mock_model)
    
    def test_predict_single_cell(self, predictor):
        """測試單一 cell 預測"""
        # 準備測試資料
        test_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        
        # 執行預測
        result = predictor.predict_single_cell(test_image)
        
        # 驗證結果
        assert isinstance(result, int)
        assert 0 <= result <= 9
    
    def test_predict_batch_cells(self, predictor):
        """測試批次預測"""
        # 準備測試資料
        test_images = [np.random.randint(0, 255, (28, 28), dtype=np.uint8) 
                      for _ in range(5)]
        
        # 執行預測
        results = predictor.predict_batch_cells(test_images)
        
        # 驗證結果
        assert len(results) == 5
        assert all(isinstance(r, int) and 0 <= r <= 9 for r in results)
    
    @pytest.mark.ai
    def test_confidence_threshold(self, predictor):
        """測試信心度門檻機制"""
        test_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        
        # 設定低信心度預測
        predictor.model.predict.return_value = np.array([[0.2, 0.15, 0.15, 0.1, 0.1,
                                                         0.1, 0.1, 0.05, 0.03, 0.02]])
        
        prediction, confidence = predictor.predict_with_confidence(test_image)
        
        assert confidence < 0.8  # 低信心度
        assert prediction == 0   # 最高機率的預測
```

### 2. 整合測試 (Integration Tests)

#### 模組間互動測試
```python
# tests/test_integration_ai_automation.py
import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.core.game_engine import GameEngine
from src.ai.predictor import Predictor
from src.automation.screen_utils import ScreenUtils

class TestAIAutomationIntegration:
    """AI 與自動化模組整合測試"""
    
    @pytest.fixture
    def game_engine(self):
        """建立遊戲引擎實例"""
        return GameEngine()
    
    @pytest.mark.integration
    def test_board_scan_to_prediction_pipeline(self, game_engine):
        """測試從盤面掃描到 AI 預測的完整管道"""
        # 模擬螢幕擷取
        with patch.object(game_engine.screen_utils, 'capture_screen') as mock_capture:
            mock_capture.return_value = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
            
            # 模擬 AI 預測
            with patch.object(game_engine.predictor, 'predict_batch_cells') as mock_predict:
                mock_predict.return_value = [1, 2, 3, 4, 5] * 5  # 25 個預測結果
                
                # 執行盤面掃描
                board_result = game_engine.scan_board()
                
                # 驗證結果
                assert board_result is not None
                assert len(board_result.flatten()) == 25
                assert all(0 <= val <= 9 for val in board_result.flatten())
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_game_detection_cycle(self, game_engine):
        """測試完整的遊戲偵測週期"""
        with patch.multiple(
            game_engine,
            detect_game_window=Mock(return_value=True),
            scan_board=Mock(return_value=np.ones((5, 5))),
            solve_board=Mock(return_value=[('move1', 'move2')]),
            execute_moves=Mock(return_value=True)
        ):
            # 執行完整週期
            result = game_engine.run_game_cycle()
            
            # 驗證各階段都被呼叫
            game_engine.detect_game_window.assert_called_once()
            game_engine.scan_board.assert_called_once()
            game_engine.solve_board.assert_called_once()
            game_engine.execute_moves.assert_called_once()
            
            assert result is True
```

### 3. 端到端測試 (E2E Tests)

#### 完整系統流程測試
```python
# tests/test_e2e_system.py
import pytest
import time
from unittest.mock import patch, Mock
from src.core.main import main_execution_loop

class TestE2ESystem:
    """端到端系統測試"""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_system_startup_and_shutdown(self):
        """測試系統啟動與關閉"""
        # 模擬系統組件
        with patch('src.core.main.initialize_system') as mock_init:
            mock_system = Mock()
            mock_init.return_value = mock_system
            
            with patch('src.core.main.cleanup_system') as mock_cleanup:
                # 模擬短暫執行後退出
                with patch('src.core.main.should_continue_running', side_effect=[True, False]):
                    
                    # 執行主程式
                    main_execution_loop()
                    
                    # 驗證初始化和清理都被呼叫
                    mock_init.assert_called_once()
                    mock_cleanup.assert_called_once()
    
    @pytest.mark.slow
    def test_error_recovery_mechanism(self):
        """測試錯誤恢復機制"""
        error_count = 0
        
        def mock_game_cycle():
            nonlocal error_count
            error_count += 1
            if error_count < 3:
                raise Exception("模擬錯誤")
            return True
        
        with patch('src.core.main.run_game_cycle', side_effect=mock_game_cycle):
            with patch('src.core.main.should_continue_after_error', return_value=True):
                with patch('src.core.main.should_continue_running', side_effect=[True, True, True, False]):
                    
                    # 執行系統，應該會重試並最終成功
                    main_execution_loop()
                    
                    # 驗證錯誤恢復機制正常運作
                    assert error_count == 3
```

### 4. 效能測試 (Performance Tests)

#### 效能基準測試
```python
# tests/test_performance.py
import pytest
import time
import psutil
import numpy as np
from src.ai.predictor import Predictor
from src.automation.screen_utils import ScreenUtils

class TestPerformance:
    """效能測試"""
    
    @pytest.mark.slow
    def test_ai_prediction_performance(self):
        """測試 AI 預測效能"""
        predictor = Predictor()
        
        # 準備測試資料
        test_images = [np.random.randint(0, 255, (28, 28), dtype=np.uint8) 
                      for _ in range(250)]  # 模擬完整盤面
        
        # 測試批次預測時間
        start_time = time.time()
        results = predictor.predict_batch_cells(test_images)
        end_time = time.time()
        
        prediction_time = end_time - start_time
        
        # 驗證效能要求
        assert prediction_time < 2.0, f"批次預測時間過長: {prediction_time:.2f}s"
        assert len(results) == 250
    
    @pytest.mark.slow
    def test_memory_usage(self):
        """測試記憶體使用"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 執行記憶體密集操作
        predictor = Predictor()
        large_batch = [np.random.randint(0, 255, (28, 28), dtype=np.uint8) 
                      for _ in range(1000)]
        
        _ = predictor.predict_batch_cells(large_batch)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 驗證記憶體使用合理
        assert memory_increase < 500, f"記憶體使用增加過多: {memory_increase:.2f}MB"
    
    def test_screen_capture_latency(self):
        """測試螢幕擷取延遲"""
        screen_utils = ScreenUtils()
        
        # 測試多次擷取的平均時間
        capture_times = []
        for _ in range(10):
            start_time = time.time()
            screenshot = screen_utils.capture_screen()
            end_time = time.time()
            
            capture_times.append(end_time - start_time)
        
        avg_capture_time = sum(capture_times) / len(capture_times)
        
        # 驗證擷取時間
        assert avg_capture_time < 0.1, f"螢幕擷取時間過長: {avg_capture_time:.3f}s"
```

## 🔧 測試工具與設定

### pytest 設定檔案
```ini
# pytest.ini
[pytest]
minversion = 6.0
addopts = 
    -ra
    --strict-markers
    --strict-config
    --cov=src
    --cov-report=term-missing:skip-covered
    --cov-report=html:htmlcov
    --cov-fail-under=80
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: 單元測試
    integration: 整合測試
    slow: 需要較長時間的測試
    ai: AI 相關測試
    automation: 自動化功能測試
    config: 配置相關測試
    visual: 視覺相關測試
```

### 測試輔助工具
```python
# tests/conftest.py
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock

@pytest.fixture(scope="session")
def test_data_dir():
    """測試資料目錄"""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def sample_cell_image():
    """範例 cell 圖像"""
    return np.random.randint(0, 255, (28, 28), dtype=np.uint8)

@pytest.fixture
def sample_board():
    """範例遊戲盤面"""
    return np.random.randint(0, 9, (5, 5))

@pytest.fixture
def mock_tensorflow_model():
    """模擬 TensorFlow 模型"""
    model = Mock()
    model.predict.return_value = np.random.rand(1, 10)
    return model

@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path, monkeypatch):
    """設定測試環境"""
    # 設定臨時目錄
    monkeypatch.setenv("MAKE10_DATA_PATH", str(tmp_path))
    
    # 建立必要的目錄結構
    (tmp_path / "models").mkdir()
    (tmp_path / "training").mkdir()
    (tmp_path / "logs").mkdir()
    
    yield tmp_path
```

## 📊 覆蓋率與品質指標

### 覆蓋率目標
```bash
# 產生覆蓋率報告
uv run pytest --cov=src --cov-report=html --cov-report=term

# 檢查覆蓋率是否達標
uv run pytest --cov=src --cov-fail-under=80
```

### 覆蓋率配置
```toml
# pyproject.toml
[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*",
    "*/conftest.py"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:"
]
```

### 品質門檻
- **單元測試覆蓋率**: > 80%
- **整合測試覆蓋率**: > 60%
- **關鍵路徑覆蓋率**: > 95%
- **測試執行時間**: < 30 秒 (不含慢速測試)

## 🚨 持續整合設定

### GitHub Actions 工作流程
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: windows-latest
    strategy:
      matrix:
        python-version: [3.12]
    
    steps:
    - uses: actions/checkout@v4
    
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
    
    - name: Run tests
      run: |
        uv run pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: uv run pytest
        language: system
        pass_filenames: false
        always_run: true
```

## 🎯 測試最佳實務

### 測試命名規範
```python
# 好的測試命名
def test_predict_single_cell_returns_integer_between_0_and_9():
    """測試單一 cell 預測返回 0-9 之間的整數"""
    pass

def test_model_loading_raises_exception_when_file_not_found():
    """測試模型載入在檔案不存在時拋出異常"""
    pass

# 避免的命名
def test_prediction():  # 太模糊
def test_1():          # 無意義
```

### 測試結構模式 (AAA)
```python
def test_predictor_batch_processing():
    """使用 Arrange-Act-Assert 模式"""
    
    # Arrange - 準備測試資料
    predictor = Predictor()
    test_images = [create_test_image() for _ in range(5)]
    expected_count = 5
    
    # Act - 執行被測試的操作
    results = predictor.predict_batch_cells(test_images)
    
    # Assert - 驗證結果
    assert len(results) == expected_count
    assert all(isinstance(r, int) for r in results)
    assert all(0 <= r <= 9 for r in results)
```

### 測試隔離原則
```python
class TestModelManager:
    """每個測試方法都應該是獨立的"""
    
    def setup_method(self):
        """每個測試前的設定"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_manager = ModelManager(self.temp_dir)
    
    def teardown_method(self):
        """每個測試後的清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_model(self):
        """測試不應該影響其他測試"""
        # 此測試的變更不會影響其他測試
        pass
```

透過遵循這個測試指南，您可以建立健全的測試套件，確保 Make10 專案的程式碼品質與系統穩定性。
