"""
測試按鍵監聽功能
"""

import sys
from pathlib import Path

# 將 src 目錄加入路徑
src_path = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(src_path))

from config.settings import cfg  # noqa: E402
from src.automation.keyboard_listener import create_keyboard_listener  # noqa: E402


def test_keyboard_listener():
    """測試按鍵監聽器的基本功能"""

    def mock_exit():
        print("模擬退出回調被呼叫")

    # 建立監聽器
    listener = create_keyboard_listener(mock_exit)

    # 測試屬性
    assert listener.exit_key == cfg.SYSTEM.exit_key.lower()
    assert listener.exit_callback == mock_exit
    assert not listener.is_running
    assert not listener.is_exit_requested()

    print(f"✅ 按鍵監聽器建立成功，退出按鍵: {listener.exit_key.upper()}")

    # 測試啟動
    listener.start()
    assert listener.is_running
    print("✅ 按鍵監聽器啟動成功")

    # 測試停止
    listener.stop()
    assert not listener.is_running
    print("✅ 按鍵監聽器停止成功")

    print("🎉 所有測試通過！")


if __name__ == "__main__":
    test_keyboard_listener()
