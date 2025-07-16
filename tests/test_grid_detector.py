"""GridDetector 類別的基礎測試"""

import numpy as np

from src.ai.grid_detector import GridDetector


class TestGridDetector:
    """GridDetector 類別測試套件"""

    def setup_method(self):
        """每個測試方法前的設定"""
        self.detector = GridDetector()

    def test_grid_detector_initialization(self):
        """測試 GridDetector 初始化"""
        assert self.detector.tolerance == 3
        assert self.detector.min_area == 20
        assert self.detector.aspect_ratio_range == (0.6, 1.4)
        assert self.detector.min_distance == 20
        assert self.detector.grid_size == (10, 25)

    def test_sample_image_colors_basic(self):
        """測試影像顏色採樣基本功能"""
        # 建立測試影像 (100x100 的藍色影像)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :] = [255, 0, 0]  # BGR 格式的藍色
        
        colors = self.detector.sample_image_colors(test_image, sample_size=10)
        
        assert len(colors) <= 10
        assert all(isinstance(color, tuple) for color in colors)
        assert all(len(color) == 3 for color in colors)

    def test_filter_valid_colors_basic(self):
        """測試顏色過濾基本功能"""
        # 模擬顏色樣本
        color_samples = [
            (255, 255, 255),  # 白色
            (255, 255, 255),  # 白色 (重複)
            (0, 0, 0),        # 黑色
            (128, 128, 128),  # 灰色
        ] * 3  # 重複以滿足 min_freq 要求
        
        valid_colors = self.detector.filter_valid_colors(
            color_samples, max_colors=5, min_freq=2
        )
        
        assert isinstance(valid_colors, list)
        assert len(valid_colors) <= 5
