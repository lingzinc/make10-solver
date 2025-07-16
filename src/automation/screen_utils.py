"""
Make10 遊戲自動化系統 - 螢幕切換工具
實作基於 PyInput 的螢幕切換功能，支援 Alt+Tab 快速切換
"""

import time

from loguru import logger
from pynput.keyboard import Controller, Key

from config.settings import cfg


def switch_screen() -> bool:
    """
    快速螢幕切換 (只按一次 Alt+Tab)

    Returns:
        bool: 切換是否成功
    """
    try:
        logger.debug("執行快速螢幕切換")

        # 建立鍵盤控制器
        keyboard = Controller()

        # 按下 Alt+Tab 並立即釋放
        keyboard.press(Key.alt)
        keyboard.press(Key.tab)
        time.sleep(cfg.AUTOMATION.screen_switch_delay)
        keyboard.release(Key.tab)
        keyboard.release(Key.alt)
        time.sleep(cfg.AUTOMATION.screen_switch_delay)

        logger.debug("快速螢幕切換完成")
        return True

    except Exception as e:
        logger.error(f"快速螢幕切換失敗: {e}")
        return False
