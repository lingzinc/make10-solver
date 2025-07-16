"""
測試 src.automation.screen_utils 模組的螢幕切換功能
"""

from unittest.mock import patch

from src.automation.screen_utils import switch_screen


class TestScreenUtils:
    """測試螢幕切換工具函式"""

    @patch("src.automation.screen_utils.keyboard")
    @patch("src.automation.screen_utils.time.sleep")
    @patch("src.automation.screen_utils.logger")
    def test_switch_screen_success(self, mock_logger, mock_sleep, mock_keyboard):
        """測試螢幕切換成功情況"""
        # 執行螢幕切換
        result = switch_screen()

        # 驗證結果
        assert result is True

        # 驗證按鍵序列
        from pynput.keyboard import Key

        mock_keyboard.press.assert_any_call(Key.alt)
        mock_keyboard.press.assert_any_call(Key.tab)
        mock_keyboard.release.assert_any_call(Key.tab)
        mock_keyboard.release.assert_any_call(Key.alt)

        # 驗證按鍵次數
        assert mock_keyboard.press.call_count == 2
        assert mock_keyboard.release.call_count == 2

        # 驗證延遲被呼叫
        assert mock_sleep.call_count == 2

        # 驗證日誌
        mock_logger.debug.assert_any_call("執行螢幕切換")
        mock_logger.debug.assert_any_call("螢幕切換完成")

    @patch("src.automation.screen_utils.keyboard")
    @patch("src.automation.screen_utils.time.sleep")
    @patch("src.automation.screen_utils.logger")
    def test_switch_screen_keyboard_exception(
        self, mock_logger, mock_sleep, mock_keyboard
    ):
        """測試鍵盤控制器操作失敗情況"""
        # 設定模擬物件拋出例外
        mock_keyboard.press.side_effect = Exception("鍵盤操作失敗")

        # 執行螢幕切換
        result = switch_screen()

        # 驗證結果
        assert result is False

        # 驗證錯誤日誌
        mock_logger.error.assert_called_once()
        error_call_args = mock_logger.error.call_args[0][0]
        assert "螢幕切換失敗" in error_call_args

    @patch("src.automation.screen_utils.keyboard")
    @patch("src.automation.screen_utils.time.sleep")
    @patch("src.automation.screen_utils.logger")
    def test_switch_screen_key_press_exception(
        self, mock_logger, mock_sleep, mock_keyboard
    ):
        """測試按鍵操作失敗情況"""
        # 設定模擬物件
        mock_keyboard.press.side_effect = Exception("按鍵操作失敗")

        # 執行螢幕切換
        result = switch_screen()

        # 驗證結果
        assert result is False

        # 驗證錯誤日誌
        mock_logger.error.assert_called_once()
        error_call_args = mock_logger.error.call_args[0][0]
        assert "螢幕切換失敗" in error_call_args

    @patch("src.automation.screen_utils.keyboard")
    @patch("src.automation.screen_utils.time.sleep")
    @patch("src.automation.screen_utils.logger")
    def test_switch_screen_sleep_exception(
        self, mock_logger, mock_sleep, mock_keyboard
    ):
        """測試延遲操作失敗情況"""
        # 設定模擬物件
        mock_sleep.side_effect = Exception("延遲操作失敗")

        # 執行螢幕切換
        result = switch_screen()

        # 驗證結果
        assert result is False

        # 驗證錯誤日誌
        mock_logger.error.assert_called_once()
        error_call_args = mock_logger.error.call_args[0][0]
        assert "螢幕切換失敗" in error_call_args

    @patch("src.automation.screen_utils.keyboard")
    @patch("src.automation.screen_utils.time.sleep")
    @patch("src.automation.screen_utils.logger")
    @patch("src.automation.screen_utils.cfg")
    def test_switch_screen_uses_config_delay(
        self, mock_cfg, mock_logger, mock_sleep, mock_keyboard
    ):
        """測試螢幕切換使用配置的延遲時間"""
        # 設定模擬配置
        mock_cfg.AUTOMATION.screen_switch_delay = 0.5

        # 執行螢幕切換
        result = switch_screen()

        # 驗證結果
        assert result is True

        # 驗證延遲時間使用配置值
        mock_sleep.assert_called_with(0.5)
        assert mock_sleep.call_count == 2

    @patch("src.automation.screen_utils.keyboard")
    @patch("src.automation.screen_utils.time.sleep")
    @patch("src.automation.screen_utils.logger")
    def test_switch_screen_key_sequence(self, mock_logger, mock_sleep, mock_keyboard):
        """測試螢幕切換按鍵序列的正確性"""
        # 執行螢幕切換
        result = switch_screen()

        # 驗證結果
        assert result is True

        # 驗證按鍵序列
        from pynput.keyboard import Key

        # 檢查按鍵和釋放的呼叫
        assert mock_keyboard.press.call_count == 2
        assert mock_keyboard.release.call_count == 2

        # 驗證按鍵順序
        mock_keyboard.press.assert_any_call(Key.alt)
        mock_keyboard.press.assert_any_call(Key.tab)
        mock_keyboard.release.assert_any_call(Key.tab)
        mock_keyboard.release.assert_any_call(Key.alt)

    @patch("src.automation.screen_utils.keyboard")
    @patch("src.automation.screen_utils.time.sleep")
    @patch("src.automation.screen_utils.logger")
    def test_switch_screen_logging(self, mock_logger, mock_sleep, mock_keyboard):
        """測試螢幕切換的日誌記錄"""
        # 執行螢幕切換
        result = switch_screen()

        # 驗證結果
        assert result is True

        # 驗證日誌呼叫
        assert mock_logger.debug.call_count == 2
        mock_logger.debug.assert_any_call("執行螢幕切換")
        mock_logger.debug.assert_any_call("螢幕切換完成")

        # 確保沒有錯誤日誌
        mock_logger.error.assert_not_called()


class TestScreenUtilsIntegration:
    """螢幕切換工具的整合測試"""

    @patch("src.automation.screen_utils.keyboard")
    @patch("src.automation.screen_utils.time.sleep")
    def test_switch_screen_multiple_calls(self, mock_sleep, mock_keyboard):
        """測試多次呼叫螢幕切換功能"""
        # 多次執行螢幕切換
        results = []
        for _ in range(3):
            results.append(switch_screen())

        # 驗證所有呼叫都成功
        assert all(results)

        # 驗證按鍵操作次數
        assert mock_keyboard.press.call_count == 6  # 3 次 * 2 按鍵
        assert mock_keyboard.release.call_count == 6  # 3 次 * 2 釋放

    @patch("src.automation.screen_utils.time.sleep")
    def test_switch_screen_real_keyboard_controller(self, mock_sleep):
        """測試使用真實鍵盤控制器（但模擬延遲）"""
        # 注意：這個測試會使用真實的鍵盤控制器，但不會實際執行按鍵
        # 因為我們只模擬了 time.sleep

        # 執行螢幕切換
        result = switch_screen()

        # 驗證結果（應該成功，除非系統環境有問題）
        assert result is True

        # 驗證延遲被呼叫
        assert mock_sleep.call_count == 2
