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
    MODEL_BATCH_SIZE,
    MODEL_CONFIDENCE_THRESHOLD,
    MODEL_VOTING_THRESHOLD,
    RETRY_ATTEMPTS,
    SCREEN_SWITCH_DELAY,
    SCREENSHOT_DELAY,
    SYSTEM_EXIT_KEY,
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
        "main_model": cfg.MODEL_DIR
        / "exports"
        / "resnet50_model.keras",  # 更新為 ResNet50 模型
        "checkpoints_dir": cfg.MODEL_DIR / "checkpoints",
        "exports_dir": cfg.MODEL_DIR / "exports",
        "resnet50_stage1": cfg.MODEL_DIR / "checkpoints" / "stage1_best.keras",
        "resnet50_stage2": cfg.MODEL_DIR / "checkpoints" / "stage2_best.keras",
    }
)

cfg.PATHS.TRAINING = EasyDict(
    {
        "images_dir": cfg.TRAINING_DIR / "images",
        "labels_dir": cfg.TRAINING_DIR / "labels",
    }
)

cfg.PATHS.ASSETS = EasyDict({"templates_dir": cfg.ASSETS_DIR / "templates"})

# 圖像處理參數 (ResNet50 版本)
cfg.IMAGE = EasyDict()
cfg.IMAGE.PROCESSING = EasyDict(
    {
        "cell_size": CELL_SIZE,  # (224, 224) for ResNet50
        "board_size": BOARD_SIZE,
        "input_channels": 3,  # RGB 三通道
        "preprocessing_method": "resnet50",  # ResNet50 預處理方法
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

# ResNet50 AI 模型參數
cfg.MODEL = EasyDict(
    {
        "input_shape": (224, 224, 3),  # ResNet50 標準輸入
        "num_classes": 10,
        "batch_size": MODEL_BATCH_SIZE,  # 16 for ResNet50
        "confidence_threshold": MODEL_CONFIDENCE_THRESHOLD,  # 0.9
        "voting_threshold": MODEL_VOTING_THRESHOLD,
        "use_pretrained": True,  # 使用 ImageNet 預訓練權重
        "fine_tune_layers": 20,  # 微調層數
        "architecture": "resnet50",  # 模型架構類型
    }
)

# ResNet50 訓練參數
cfg.TRAINING = EasyDict(
    {
        "epochs_stage1": 10,  # 第一階段訓練輪數 (凍結預訓練層)
        "epochs_stage2": 20,  # 第二階段訓練輪數 (微調)
        "learning_rate_stage1": 0.001,  # 第一階段學習率
        "learning_rate_stage2": 0.0001,  # 第二階段學習率 (較小)
        "validation_split": 0.2,  # 驗證集比例
        "early_stopping_patience": 15,  # 早停忍耐度
        "reduce_lr_patience": 5,  # 學習率衰減忍耐度
        "reduce_lr_factor": 0.5,  # 學習率衰減倍數
    }
)

# ResNet50 資料擴增參數
cfg.DATA_AUGMENTATION = EasyDict(
    {
        "rotation_range": 10,  # 旋轉角度範圍
        "width_shift_range": 0.1,  # 水平移動範圍
        "height_shift_range": 0.1,  # 垂直移動範圍
        "zoom_range": 0.1,  # 縮放範圍
        "brightness_range": [0.8, 1.2],  # 亮度調整範圍
        "horizontal_flip": False,  # 不水平翻轉 (數字會變形)
        "vertical_flip": False,  # 不垂直翻轉
        "fill_mode": "constant",  # 填充模式
        "cval": 0.0,  # 填充值
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

# 系統控制參數
cfg.SYSTEM = EasyDict(
    {
        "exit_key": SYSTEM_EXIT_KEY,
    }
)

# 螢幕和遊戲檢測參數
cfg.GAME = EasyDict(
    {
        "template_match_threshold": TEMPLATE_MATCH_THRESHOLD,
        "screen_capture_region": None,  # None 表示全螢幕
        "reset_button_template": ["reset_button_b.png", "reset_button_w.png"],
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
        "model_config": cfg.MODEL,
        "automation_config": cfg.AUTOMATION,
        "game_detection": cfg.GAME,
        "debug_config": cfg.DEBUG,
    }
