"""
Make10 遊戲自動化系統 - 按鍵監聽模組
實作系統按鍵監聽和優雅關閉功能
"""

import threading
from collections.abc import Callable

from loguru import logger
from pynput import keyboard

from config.settings import cfg


class KeyboardListener:
    """系統按鍵監聽器"""

    def __init__(self, exit_callback: Callable | None = None):
        """
        初始化按鍵監聽器

        Args:
            exit_callback: 當按下退出按鍵時呼叫的回調函式
        """
        self.exit_callback = exit_callback
        self.exit_key = cfg.SYSTEM.exit_key.lower()
        self.listener = None
        self.is_running = False
        self._stop_event = threading.Event()

    def on_press(self, key):
        """
        按鍵按下事件處理

        Args:
            key: 按下的按鍵
        """
        try:
            # 處理功能鍵
            if hasattr(key, "name"):
                key_name = key.name.lower()
            else:
                # 處理普通字元鍵
                key_name = str(key).replace("'", "").lower()

            logger.debug(f"偵測到按鍵: {key_name}")

            # 檢查是否為退出按鍵
            if key_name == self.exit_key:
                logger.info(
                    f"偵測到退出按鍵 [{self.exit_key.upper()}]，正在優雅關閉系統..."
                )
                self._stop_event.set()

                # 呼叫退出回調函式
                if self.exit_callback:
                    self.exit_callback()

                # 停止監聽器
                return False

        except Exception as e:
            logger.error(f"按鍵處理錯誤: {e}")

    def start(self):
        """啟動按鍵監聽"""
        if self.is_running:
            logger.warning("按鍵監聽器已在執行中")
            return

        try:
            logger.info(f"啟動按鍵監聽器，退出按鍵: {self.exit_key.upper()}")

            # 建立並啟動監聽器
            self.listener = keyboard.Listener(on_press=self.on_press)

            self.listener.start()
            self.is_running = True

            logger.debug("按鍵監聽器已啟動")

        except Exception as e:
            logger.error(f"啟動按鍵監聽器失敗: {e}")
            self.is_running = False

    def stop(self):
        """停止按鍵監聽"""
        if not self.is_running:
            return

        try:
            logger.debug("停止按鍵監聽器...")

            if self.listener:
                self.listener.stop()

            self.is_running = False
            self._stop_event.set()

            logger.debug("按鍵監聽器已停止")

        except Exception as e:
            logger.error(f"停止按鍵監聽器失敗: {e}")

    def wait_for_exit(self, timeout: float | None = None):
        """
        等待退出事件

        Args:
            timeout: 等待超時時間（秒），None 表示無限等待

        Returns:
            bool: 是否因為退出事件而返回
        """
        return self._stop_event.wait(timeout)

    def is_exit_requested(self) -> bool:
        """
        檢查是否已請求退出

        Returns:
            bool: 是否已請求退出
        """
        return self._stop_event.is_set()


def create_keyboard_listener(exit_callback: Callable | None = None) -> KeyboardListener:
    """
    建立按鍵監聽器的工廠函式

    Args:
        exit_callback: 退出回調函式

    Returns:
        KeyboardListener: 按鍵監聽器實例
    """
    return KeyboardListener(exit_callback)
