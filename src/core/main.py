"""Make10 遊戲自動化系統主程式入口"""

import time

from loguru import logger

from automation.keyboard_listener import create_keyboard_listener
from automation.screen_utils import (
    click_at_position,
    find_reset_button,
    switch_screen,
)
from config.settings import cfg


class GameAutomationSystem:
    """Make10 遊戲自動化系統核心類別"""

    def __init__(self):
        self.initialized = False
        self.keyboard_listener = None
        self.should_exit = False

    def request_exit(self):
        """請求系統退出"""
        logger.info("收到退出請求，準備停止系統...")
        self.should_exit = True

    def initialize(self):
        """初始化系統核心模組"""
        logger.info("初始化 Make10 遊戲自動化系統...")
        logger.debug("載入系統配置...")

        # 初始化按鍵監聽器
        self.keyboard_listener = create_keyboard_listener(self.request_exit)
        self.keyboard_listener.start()

        # TODO: 初始化 AI 模型
        # TODO: 初始化電腦視覺模組
        # TODO: 初始化自動化控制模組
        self.initialized = True
        logger.info("系統核心模組初始化完成")

    def _detect_game_board(self):
        """偵測遊戲盤面 - 私有方法"""
        from automation.screen_utils import capture_screen

        logger.info("開始偵測遊戲盤面...")
        screen_image = capture_screen()

        if screen_image is None:
            logger.error("無法截取螢幕畫面")
            return None

        return self._process_board_detection(screen_image)

    def _process_board_detection(self, screen_image):
        """處理盤面偵測邏輯 - 私有方法"""
        try:
            # 初始化方格偵測器
            from ai.grid_detector import GridDetector

            grid_detector = GridDetector()

            # 偵測盤面資訊
            board_info = grid_detector.detect_board_from_screen_image(
                screen_image, enable_validation=False
            )

            if board_info["success"]:
                # TODO: 將盤面資訊傳遞給後續的數字識別模組
                # TODO: 實作數字識別邏輯
                # TODO: 實作遊戲求解演算法
                return board_info
            else:
                logger.error(f"❌ {board_info['message']}")
                return None

        except Exception as e:
            logger.error(f"盤面偵測過程發生錯誤: {e}")
            return None

    def run_game_loop(self):
        """執行主要遊戲自動化流程（單次執行）"""
        if not self.initialized:
            raise RuntimeError("系統尚未初始化")

        logger.info("開始遊戲自動化流程")

        # 實作遊戲檢測邏輯
        logger.info("切換到遊戲畫面")
        if not switch_screen():
            logger.error("畫面切換失敗")
            return

        # 等待畫面切換完成
        time.sleep(cfg.AUTOMATION.screen_switch_delay)

        logger.info("搜尋 reset 按鈕")
        reset_position = find_reset_button()

        # 早期返回模式，減少巢狀結構
        if not reset_position:
            logger.warning("未找到 reset 按鈕")
            return

        x, y = reset_position
        logger.info(f"找到 reset 按鈕，座標: ({x}, {y})")

        # 點擊 reset 按鈕
        if not click_at_position(x, y):
            logger.error("點擊 reset 按鈕失敗")
            return

        logger.info("成功點擊 reset 按鈕")
        time.sleep(0.5)  # 等待按鈕響應

        # 偵測遊戲盤面
        self._detect_game_board()

        switch_screen()
        time.sleep(cfg.AUTOMATION.screen_switch_delay)

        # TODO: 實作數字識別和計算邏輯
        # TODO: 實作自動點擊和拖拉邏輯
        # TODO: 實作遊戲狀態監控

        # 單次執行遊戲處理邏輯
        try:
            logger.info("執行遊戲處理邏輯...")
            # 模擬遊戲處理邏輯
            time.sleep(1.0)
            logger.info("遊戲處理邏輯執行完成")

        except KeyboardInterrupt:
            logger.info("收到中斷信號")
            self.should_exit = True

    def shutdown(self):
        """系統關閉清理"""
        logger.info("系統正在關閉...")

        # 停止按鍵監聽器
        if self.keyboard_listener:
            self.keyboard_listener.stop()

        # TODO: 清理 AI 模型資源
        # TODO: 清理視覺處理資源
        # TODO: 停止自動化控制
        self.initialized = False
        logger.info("系統資源清理完成")


def main():
    """主程式入口函式"""
    system = GameAutomationSystem()

    try:
        system.initialize()
        system.run_game_loop()
    except KeyboardInterrupt:
        logger.info("使用者中斷程式執行")
    except Exception as e:
        logger.error(f"系統執行錯誤: {e}")
        raise
    finally:
        system.shutdown()


if __name__ == "__main__":
    main()
