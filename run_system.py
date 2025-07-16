"""Make10 遊戲自動化系統 - 系統執行入口"""

import sys
from pathlib import Path

from loguru import logger

# 將 src 目錄加入 Python 路徑
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.core.main import main  # noqa: E402


def setup_logging() -> None:
    """設定系統日誌"""
    # 建立 logs 目錄
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # 移除預設處理器，避免重複輸出
    logger.remove()

    # 添加終端處理器
    logger.add(
        sys.stderr,
        level="DEBUG",
        format="<green>{time:YY-MM-DD HH:mm:ss}</green> <level>[{level}]</level> <cyan>{message}</cyan>",
        colorize=True,
    )

    # 設定日誌檔案
    logger.add(
        "logs/make10_system.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
        format="{time:YY-MM-DD HH:mm:ss} [{level}] {message}",
    )


def check_dependencies() -> bool:
    """檢查系統相依性"""
    required_packages = {
        "cv2": "opencv-python",
        "numpy": "numpy",
        "pynput": "pynput",
        "loguru": "loguru",
        "easydict": "easydict",
    }

    missing_packages = []

    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        logger.error(f"缺少必要套件: {', '.join(missing_packages)}")
        logger.info("請執行: pip install " + " ".join(missing_packages))
        return False

    logger.info("相依套件檢查通過")
    return True


def run_system():
    """系統啟動入口"""
    print("=== Make10 遊戲自動化系統 v0.1.0 ===")

    # 設定日誌
    setup_logging()
    logger.info("系統啟動中...")

    # 檢查相依性
    if not check_dependencies():
        logger.error("系統相依性檢查失敗，無法啟動")
        return False

    # 啟動主程式
    try:
        main()
        return True
    except Exception as e:
        logger.error(f"系統啟動失敗: {e}")
        return False


if __name__ == "__main__":
    success = run_system()
    sys.exit(0 if success else 1)
