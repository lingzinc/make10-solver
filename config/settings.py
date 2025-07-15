"""
Make10 遊戲自動化系統 - 配置設定
包含路徑、參數配置和系統設定
"""

from pathlib import Path

from easydict import EasyDict

from .constants import (
    AUTOMATION_TIMEOUT,
    BOARD_SIZE,
    CANNY_THRESHOLD1,
    CANNY_THRESHOLD2,
    CELL_SIZE,
    CLICK_DELAY,
    DRAG_DELAY,
    HOUGH_MAX_LINE_GAP,
    HOUGH_MIN_LINE_LENGTH,
    HOUGH_RHO,
    HOUGH_THETA_RESOLUTION,
    HOUGH_THRESHOLD,
    MODEL_BATCH_SIZE,
    MODEL_CONFIDENCE_THRESHOLD,
    MODEL_VOTING_THRESHOLD,
    RETRY_ATTEMPTS,
    SCREEN_SWITCH_DELAY,
    SCREENSHOT_DELAY,
    TEMPLATE_MATCH_THRESHOLD,
)

# 建立配置物件
cfg = EasyDict()

# 專案根目錄
cfg.PROJECT_ROOT = Path(__file__).parents[1]
cfg.DATA_DIR = cfg.PROJECT_ROOT / "data"
cfg.MODEL_DIR = cfg.DATA_DIR / "models"
cfg.TRAINING_DIR = cfg.DATA_DIR / "training"
cfg.ASSETS_DIR = cfg.DATA_DIR / "assets"

# 路徑配置
cfg.PATHS = EasyDict()
cfg.PATHS.MODEL = EasyDict(
    {
        "main_model": cfg.MODEL_DIR / "exports" / "model.keras",
        "checkpoints_dir": cfg.MODEL_DIR / "checkpoints",
        "exports_dir": cfg.MODEL_DIR / "exports",
    }
)

cfg.PATHS.TRAINING = EasyDict(
    {
        "images_dir": cfg.TRAINING_DIR / "images",
        "labels_dir": cfg.TRAINING_DIR / "labels",
    }
)

cfg.PATHS.ASSETS = EasyDict({"templates_dir": cfg.ASSETS_DIR / "templates"})

# 圖像處理參數
cfg.IMAGE = EasyDict()
cfg.IMAGE.PROCESSING = EasyDict(
    {
        "cell_size": CELL_SIZE,
        "board_size": BOARD_SIZE,
        "preprocessing": EasyDict(
            {
                "gaussian_blur_kernel": (3, 3),
                "canny_threshold1": CANNY_THRESHOLD1,
                "canny_threshold2": CANNY_THRESHOLD2,
                "morph_kernel_size": (3, 3),
            }
        ),
    }
)

# 霍夫直線檢測參數
cfg.HOUGH = EasyDict(
    {
        "rho": HOUGH_RHO,
        "theta_resolution": HOUGH_THETA_RESOLUTION,
        "threshold": HOUGH_THRESHOLD,
        "min_line_length": HOUGH_MIN_LINE_LENGTH,
        "max_line_gap": HOUGH_MAX_LINE_GAP,
    }
)

# AI 模型參數
cfg.MODEL = EasyDict(
    {
        "input_shape": (28, 28, 1),
        "num_classes": 10,
        "batch_size": MODEL_BATCH_SIZE,
        "confidence_threshold": MODEL_CONFIDENCE_THRESHOLD,
        "voting_threshold": MODEL_VOTING_THRESHOLD,
    }
)

# 自動化控制參數
cfg.AUTOMATION = EasyDict(
    {
        "click_delay": CLICK_DELAY,
        "drag_delay": DRAG_DELAY,
        "screen_switch_delay": SCREEN_SWITCH_DELAY,
        "screenshot_delay": SCREENSHOT_DELAY,
        "retry_attempts": RETRY_ATTEMPTS,
        "timeout": AUTOMATION_TIMEOUT,
    }
)

# 螢幕和遊戲檢測參數
cfg.GAME = EasyDict(
    {
        "template_match_threshold": TEMPLATE_MATCH_THRESHOLD,
        "screen_capture_region": None,  # None 表示全螢幕
        "reset_button_template": "resetButton.png",
    }
)

# 除錯和日誌設定
cfg.DEBUG = EasyDict(
    {
        "enable_debug": True,
        "save_debug_images": True,
        "debug_output_dir": cfg.PROJECT_ROOT / "debug_output",
        "log_level": "INFO",
        "performance_monitoring": True,
    }
)


# 驗證設定
def validate_paths() -> bool:
    """驗證所有必要的路徑是否存在"""
    required_dirs = [cfg.DATA_DIR, cfg.MODEL_DIR, cfg.TRAINING_DIR, cfg.ASSETS_DIR]

    for path in required_dirs:
        if not path.exists():
            print(f"警告: 路徑不存在 - {path}")
            return False

    return True


def get_model_path() -> Path:
    """取得主要模型檔案路徑"""
    model_path = cfg.PATHS.MODEL.main_model
    if not model_path.exists():
        raise FileNotFoundError(f"模型檔案不存在: {model_path}")
    return model_path


def create_debug_dir() -> None:
    """建立除錯輸出目錄"""
    if cfg.DEBUG.save_debug_images:
        debug_dir = cfg.DEBUG.debug_output_dir
        debug_dir.mkdir(parents=True, exist_ok=True)


# 初始化設定
def initialize_settings() -> dict[str, any]:
    """初始化所有設定並驗證"""
    # 驗證路徑
    if not validate_paths():
        print("路徑驗證失敗，部分功能可能無法正常運作")

    # 建立除錯目錄
    create_debug_dir()

    return {
        "paths": {
            "project_root": cfg.PROJECT_ROOT,
            "data_dir": cfg.DATA_DIR,
            "model_paths": cfg.PATHS.MODEL,
            "training_paths": cfg.PATHS.TRAINING,
            "asset_paths": cfg.PATHS.ASSETS,
        },
        "image_processing": cfg.IMAGE.PROCESSING,
        "hough_params": cfg.HOUGH,
        "model_config": cfg.MODEL,
        "automation_config": cfg.AUTOMATION,
        "game_detection": cfg.GAME,
        "debug_config": cfg.DEBUG,
    }
