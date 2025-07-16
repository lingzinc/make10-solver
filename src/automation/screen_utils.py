"""
Make10 遊戲自動化系統 - 螢幕工具
"""

import time
from pathlib import Path

import cv2
import mss
import numpy as np
from loguru import logger
from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController

from config.settings import cfg

# 建立鍵盤、滑鼠控制器
keyboard = KeyboardController()
mouse = MouseController()


def switch_screen() -> bool:
    """
    螢幕切換 (Alt+Tab)

    Returns:
        bool: 切換是否成功
    """
    try:
        logger.debug("執行螢幕切換")

        # 按下 Alt+Tab 並立即釋放
        keyboard.press(Key.alt)
        keyboard.press(Key.tab)
        time.sleep(cfg.AUTOMATION.screen_switch_delay)
        keyboard.release(Key.tab)
        keyboard.release(Key.alt)
        time.sleep(cfg.AUTOMATION.screen_switch_delay)

        logger.debug("螢幕切換完成")
        return True

    except Exception as e:
        logger.error(f"螢幕切換失敗: {e}")
        return False


def capture_screen() -> np.ndarray | None:
    """
    截取當前螢幕畫面

    Returns:
        np.ndarray | None: 截圖的 OpenCV 格式圖像，失敗時返回 None
    """
    try:
        logger.debug("開始截取螢幕畫面")

        # 等待螢幕穩定
        time.sleep(cfg.AUTOMATION.screenshot_delay)

        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            img_array = np.array(screenshot)
            screenshot_cv = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2BGR)

        logger.debug(f"螢幕截圖成功，尺寸: {screenshot_cv.shape}")
        return screenshot_cv

    except Exception as e:
        logger.error(f"螢幕截圖失敗: {e}")
        return None


def find_template_in_screen(
    template_path: str, threshold: float = None
) -> tuple[int, int] | None:
    """
    在螢幕截圖中尋找模板圖像

    Args:
        template_path (str): 模板圖像檔案路徑
        threshold (float): 相似度閾值，None 時使用設定檔預設值

    Returns:
        tuple[int, int] | None: 找到的圖像中心座標 (x, y)，找不到時返回 None
    """
    try:
        # 使用設定檔中的預設閾值
        if threshold is None:
            threshold = cfg.GAME.template_match_threshold
        # 檢查模板檔案是否存在
        if not Path(template_path).exists():
            logger.error(f"模板檔案不存在: {template_path}")
            return None

        # 載入模板圖像
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if template is None:
            logger.error(f"無法載入模板圖像: {template_path}")
            return None

        # 截取螢幕
        screenshot = capture_screen()
        if screenshot is None:
            return None

        # 進行模板匹配
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        logger.debug(f"模板匹配結果 - 最大相似度: {max_val:.3f}, 閾值: {threshold}")

        # 檢查相似度是否達到閾值
        if max_val >= threshold:
            # 計算圖像中心座標
            template_h, template_w = template.shape[:2]
            center_x = max_loc[0] + template_w // 2
            center_y = max_loc[1] + template_h // 2

            logger.info(
                f"找到模板圖像: {Path(template_path).name}, 座標: ({center_x}, {center_y}), 相似度: {max_val:.3f}"
            )
            return (center_x, center_y)
        else:
            logger.debug(
                f"模板圖像相似度不足: {Path(template_path).name}, 相似度: {max_val:.3f}"
            )
            return None

    except Exception as e:
        logger.error(f"模板匹配失敗: {e}")
        return None


def find_reset_button() -> tuple[int, int] | None:
    """
    尋找 reset 按鈕位置

    Returns:
        Optional[Tuple[int, int]]: reset 按鈕的座標 (x, y)，找不到時返回 None
    """
    # 模板圖像路徑
    template_dir = Path(__file__).parent.parent.parent / "data" / "assets" / "templates"
    templates = [
        template_dir / "reset_button_w.png",  # 先嘗試白色按鈕
        template_dir / "reset_button_b.png",  # 再嘗試黑色按鈕
    ]

    logger.info("開始尋找 reset 按鈕...")

    for template_path in templates:
        logger.debug(f"嘗試匹配模板: {template_path.name}")
        position = find_template_in_screen(str(template_path))

        if position:
            logger.info(f"成功找到 reset 按鈕 ({template_path.name})")
            return position

    logger.warning("未找到 reset 按鈕")
    return None


def click_at_position(x: int, y: int) -> bool:
    """
    在指定座標點擊滑鼠左鍵

    Args:
        x (int): X 座標
        y (int): Y 座標

    Returns:
        bool: 點擊是否成功
    """
    try:
        logger.debug(f"移動滑鼠至座標: ({x}, {y})")
        mouse.position = (x, y)
        time.sleep(cfg.AUTOMATION.click_delay)
        mouse.click(Button.left, 1)
        time.sleep(cfg.AUTOMATION.click_delay)

        logger.info(f"成功點擊座標: ({x}, {y})")
        return True

    except Exception as e:
        logger.error(f"點擊失敗: {e}")
        return False
