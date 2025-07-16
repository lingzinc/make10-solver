"""
測試 config.settings 模組的配置功能
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from config.settings import (
    cfg,
    create_debug_dir,
    get_model_path,
    initialize_settings,
    validate_paths,
)


class TestConfigurationObject:
    """測試配置物件的結構和內容"""

    def test_cfg_structure(self):
        """測試 cfg 物件基本結構存在"""
        assert hasattr(cfg, "PROJECT_ROOT")
        assert hasattr(cfg, "DATA_DIR")
        assert hasattr(cfg, "MODEL_DIR")
        assert hasattr(cfg, "TRAINING_DIR")
        assert hasattr(cfg, "ASSETS_DIR")
        assert hasattr(cfg, "PATHS")
        assert hasattr(cfg, "IMAGE")
        assert hasattr(cfg, "MODEL")
        assert hasattr(cfg, "AUTOMATION")
        assert hasattr(cfg, "GAME")
        assert hasattr(cfg, "DEBUG")

    def test_paths_structure(self):
        """測試路徑配置結構"""
        assert hasattr(cfg.PATHS, "MODEL")
        assert hasattr(cfg.PATHS, "TRAINING")
        assert hasattr(cfg.PATHS, "ASSETS")

        # 檢查模型路徑
        assert hasattr(cfg.PATHS.MODEL, "main_model")
        assert hasattr(cfg.PATHS.MODEL, "checkpoints_dir")
        assert hasattr(cfg.PATHS.MODEL, "exports_dir")

        # 檢查訓練路徑
        assert hasattr(cfg.PATHS.TRAINING, "images_dir")
        assert hasattr(cfg.PATHS.TRAINING, "labels_dir")

        # 檢查資源路徑
        assert hasattr(cfg.PATHS.ASSETS, "templates_dir")

    def test_image_processing_structure(self):
        """測試圖像處理配置結構"""
        assert hasattr(cfg.IMAGE, "PROCESSING")
        processing = cfg.IMAGE.PROCESSING

        assert "cell_size" in processing
        assert "board_size" in processing
        assert "preprocessing" in processing

        preprocessing = processing.preprocessing
        assert "gaussian_blur_kernel" in preprocessing
        assert "canny_threshold1" in preprocessing
        assert "canny_threshold2" in preprocessing
        assert "morph_kernel_size" in preprocessing

    def test_model_parameters_structure(self):
        """測試 AI 模型參數結構"""
        model = cfg.MODEL

        required_params = [
            "input_shape",
            "num_classes",
            "batch_size",
            "confidence_threshold",
            "voting_threshold",
        ]

        for param in required_params:
            assert param in model

    def test_automation_parameters_structure(self):
        """測試自動化參數結構"""
        automation = cfg.AUTOMATION

        required_params = [
            "click_delay",
            "drag_delay",
            "screen_switch_delay",
            "screenshot_delay",
            "retry_attempts",
            "timeout",
        ]

        for param in required_params:
            assert param in automation

    def test_game_parameters_structure(self):
        """測試遊戲檢測參數結構"""
        game = cfg.GAME

        required_params = [
            "template_match_threshold",
            "screen_capture_region",
            "reset_button_template",
        ]

        for param in required_params:
            assert param in game

    def test_debug_parameters_structure(self):
        """測試除錯參數結構"""
        debug = cfg.DEBUG

        required_params = [
            "enable_debug",
            "save_debug_images",
            "debug_output_dir",
            "log_level",
            "performance_monitoring",
        ]

        for param in required_params:
            assert param in debug


class TestPathValidation:
    """測試路徑相關功能"""

    def test_path_types(self):
        """測試路徑類型正確"""
        assert isinstance(cfg.PROJECT_ROOT, Path)
        assert isinstance(cfg.DATA_DIR, Path)
        assert isinstance(cfg.MODEL_DIR, Path)
        assert isinstance(cfg.TRAINING_DIR, Path)
        assert isinstance(cfg.ASSETS_DIR, Path)

    def test_path_relationships(self):
        """測試路徑關係正確"""
        # 檢查相對路徑關係
        assert cfg.DATA_DIR == cfg.PROJECT_ROOT / "data"
        assert cfg.MODEL_DIR == cfg.DATA_DIR / "models"
        assert cfg.TRAINING_DIR == cfg.DATA_DIR / "training"
        assert cfg.ASSETS_DIR == cfg.DATA_DIR / "assets"

    @patch("pathlib.Path.exists")
    def test_validate_paths_success(self, mock_exists):
        """測試路徑驗證成功"""
        mock_exists.return_value = True
        assert validate_paths() is True

    @patch("pathlib.Path.exists")
    @patch("builtins.print")
    def test_validate_paths_failure(self, mock_print, mock_exists):
        """測試路徑驗證失敗"""
        mock_exists.return_value = False
        assert validate_paths() is False
        assert mock_print.called


class TestModelPath:
    """測試模型路徑功能"""

    @patch("pathlib.Path.exists")
    def test_get_model_path_success(self, mock_exists):
        """測試成功獲取模型路徑"""
        mock_exists.return_value = True
        model_path = get_model_path()
        assert isinstance(model_path, Path)
        assert model_path == cfg.PATHS.MODEL.main_model

    @patch("pathlib.Path.exists")
    def test_get_model_path_file_not_found(self, mock_exists):
        """測試模型檔案不存在"""
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError):
            get_model_path()


class TestDebugDirectory:
    """測試除錯目錄功能"""

    @patch("pathlib.Path.mkdir")
    def test_create_debug_dir_enabled(self, mock_mkdir):
        """測試除錯目錄建立（啟用時）"""
        original_save_debug = cfg.DEBUG.save_debug_images
        cfg.DEBUG.save_debug_images = True

        create_debug_dir()

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        cfg.DEBUG.save_debug_images = original_save_debug

    @patch("pathlib.Path.mkdir")
    def test_create_debug_dir_disabled(self, mock_mkdir):
        """測試除錯目錄建立（停用時）"""
        original_save_debug = cfg.DEBUG.save_debug_images
        cfg.DEBUG.save_debug_images = False

        create_debug_dir()

        mock_mkdir.assert_not_called()
        cfg.DEBUG.save_debug_images = original_save_debug


class TestInitializeSettings:
    """測試設定初始化功能"""

    @patch("config.settings.validate_paths")
    @patch("config.settings.create_debug_dir")
    def test_initialize_settings_success(self, mock_create_debug, mock_validate):
        """測試設定初始化成功"""
        mock_validate.return_value = True

        result = initialize_settings()

        assert isinstance(result, dict)
        assert "paths" in result
        assert "image_processing" in result
        assert "model_config" in result
        assert "automation_config" in result
        assert "game_detection" in result
        assert "debug_config" in result

        mock_validate.assert_called_once()
        mock_create_debug.assert_called_once()

    @patch("config.settings.validate_paths")
    @patch("config.settings.create_debug_dir")
    @patch("builtins.print")
    def test_initialize_settings_validation_failure(
        self, mock_print, mock_create_debug, mock_validate
    ):
        """測試設定初始化路徑驗證失敗"""
        mock_validate.return_value = False

        result = initialize_settings()

        assert isinstance(result, dict)
        mock_print.assert_called_once()
        mock_validate.assert_called_once()
        mock_create_debug.assert_called_once()


class TestConfigurationValues:
    """測試配置值的正確性"""

    def test_image_processing_values(self):
        """測試圖像處理參數值"""
        processing = cfg.IMAGE.PROCESSING

        # 檢查基本參數
        assert isinstance(processing.cell_size, tuple)
        assert isinstance(processing.board_size, tuple)

        # 檢查預處理參數
        preprocessing = processing.preprocessing
        assert isinstance(preprocessing.gaussian_blur_kernel, tuple)
        assert isinstance(preprocessing.canny_threshold1, int)
        assert isinstance(preprocessing.canny_threshold2, int)
        assert isinstance(preprocessing.morph_kernel_size, tuple)

    def test_model_values(self):
        """測試模型參數值"""
        model = cfg.MODEL

        assert model.input_shape == (28, 28, 1)
        assert model.num_classes == 10
        assert isinstance(model.batch_size, int)
        assert 0.0 <= model.confidence_threshold <= 1.0
        assert 0.0 <= model.voting_threshold <= 1.0

    def test_automation_values(self):
        """測試自動化參數值"""
        automation = cfg.AUTOMATION

        for param in [
            "click_delay",
            "drag_delay",
            "screen_switch_delay",
            "screenshot_delay",
        ]:
            value = getattr(automation, param)
            assert isinstance(value, int | float)
            assert value >= 0

    def test_debug_values(self):
        """測試除錯參數值"""
        debug = cfg.DEBUG

        assert isinstance(debug.enable_debug, bool)
        assert isinstance(debug.save_debug_images, bool)
        assert isinstance(debug.debug_output_dir, Path)
        assert isinstance(debug.log_level, str)
        assert isinstance(debug.performance_monitoring, bool)
