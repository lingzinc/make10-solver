"""
Make10 遊戲 - 方格盤面偵測模組
基於 grid_detector_v7.py 整合版本
"""

from collections import defaultdict

import cv2
import numpy as np
from loguru import logger


class GridDetector:
    """Make10 遊戲方格盤面偵測器"""

    def __init__(self):
        self.tolerance = 3
        self.min_area = 20
        self.aspect_ratio_range = (0.6, 1.4)
        self.min_distance = 20
        self.grid_size = (10, 25)  # rows, cols

    def sample_image_colors(
        self, image: np.ndarray, sample_size: int = 1000
    ) -> list[tuple]:
        """從影像中採樣顏色"""
        small_image = cv2.resize(
            image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST
        )
        small_rgb = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)

        # 隨機採樣
        np.random.seed(42)
        h, w = small_rgb.shape[:2]
        sample_points = np.random.randint(0, min(h, w), size=(sample_size, 2))

        color_samples = []
        for y, x in sample_points:
            if y < h and x < w:
                color_samples.append(tuple(small_rgb[y, x]))

        return color_samples

    def filter_valid_colors(
        self,
        color_samples: list[tuple],
        max_colors: int = 15,
        min_freq: int = 2,
        color_range: tuple[int, int] = (100, 250),
    ) -> list[np.ndarray]:
        """篩選有效的方格顏色"""
        color_freq = defaultdict(int)
        for color in color_samples:
            color_freq[color] += 1

        colors = []
        min_val, max_val = color_range

        for (r, g, b), freq in sorted(
            color_freq.items(), key=lambda x: x[1], reverse=True
        ):
            if (
                freq > min_freq
                and r != 255
                and g != 255
                and b != 255
                and r > 0
                and g > 0
                and b > 0
                and min_val <= r <= max_val
                and min_val <= g <= max_val
                and min_val <= b <= max_val
            ):
                colors.append(np.array([r, g, b], dtype=np.uint8))
                if len(colors) >= max_colors:
                    break

        return colors

    def get_fallback_colors(self) -> list[np.ndarray]:
        """取得預設的方格顏色"""
        return [
            np.array([235, 230, 219], dtype=np.uint8),
            np.array([224, 217, 209], dtype=np.uint8),
            np.array([209, 217, 224], dtype=np.uint8),
            np.array([224, 219, 199], dtype=np.uint8),
            np.array([219, 199, 194], dtype=np.uint8),
            np.array([199, 214, 199], dtype=np.uint8),
            np.array([209, 199, 214], dtype=np.uint8),
            np.array([184, 173, 194], dtype=np.uint8),
            np.array([173, 163, 148], dtype=np.uint8),
        ]

    def auto_detect_grid_colors(
        self,
        image: np.ndarray,
        sample_size: int = 1000,
        max_colors: int = 15,
        color_range: tuple[int, int] = (100, 250),
    ) -> list[np.ndarray]:
        """自動偵測盤面中的方格顏色"""
        try:
            color_samples = self.sample_image_colors(image, sample_size)
            colors = self.filter_valid_colors(
                color_samples, max_colors, color_range=color_range
            )

            # 如果沒有找到有效顏色，使用預設顏色
            if not colors:
                colors = self.get_fallback_colors()

            return colors

        except Exception as e:
            logger.warning(f"自動顏色偵測失敗，使用預設顏色: {e}")
            return self.get_fallback_colors()

    def extract_contour_centers(
        self,
        image_rgb: np.ndarray,
        color: np.ndarray,
    ) -> list[tuple[int, int]]:
        """從單一顏色中提取輪廓中心"""
        # 顏色容差匹配
        mask = cv2.inRange(
            image_rgb,
            np.maximum(color - self.tolerance, 0),
            np.minimum(color + self.tolerance, 255),
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centers = []
        min_ratio, max_ratio = self.aspect_ratio_range

        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                if min_ratio <= w / h <= max_ratio:
                    centers.append((int((x + w // 2) * 2), int((y + h // 2) * 2)))

        return centers

    def remove_duplicate_centers_vectorized(
        self, centers: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """向量化去重函式"""
        if len(centers) <= 1:
            return centers

        centers_array = np.array(centers)
        unique_centers = []
        used = np.zeros(len(centers), dtype=bool)

        distance_threshold = self.min_distance**2  # 避免開根號計算

        for i, center in enumerate(centers):
            if used[i]:
                continue
            unique_centers.append(center)
            distances_sq = np.sum((centers_array - center) ** 2, axis=1)
            used |= distances_sq < distance_threshold

        return unique_centers

    def detect_colored_grids_auto(self, image: np.ndarray) -> list[tuple[int, int]]:
        """使用自動偵測的顏色來尋找方格"""
        colors = self.auto_detect_grid_colors(image)

        small_image = cv2.resize(
            image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST
        )
        small_rgb = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)

        centers = []
        for color in colors:
            color_centers = self.extract_contour_centers(small_rgb, color)
            centers.extend(color_centers)

        # 去重
        if len(centers) > 1:
            centers = self.remove_duplicate_centers_vectorized(centers)

        return centers

    def calculate_grid_distances(
        self, points: np.ndarray
    ) -> tuple[list[float], list[float]]:
        """計算方格間距離"""
        # 向量化距離計算
        diffs = points[:, np.newaxis] - points[np.newaxis, :]
        distances = np.abs(diffs)

        # 水平和垂直距離篩選
        mask_h = (
            (distances[:, :, 1] < 30)
            & (distances[:, :, 0] > 10)
            & (distances[:, :, 0] < 200)
        )
        mask_v = (
            (distances[:, :, 0] < 30)
            & (distances[:, :, 1] > 10)
            & (distances[:, :, 1] < 200)
        )

        distances_x = distances[:, :, 0][mask_h].tolist()
        distances_y = distances[:, :, 1][mask_v].tolist()

        return distances_x, distances_y

    def get_common_distance(self, distances: list[float]) -> float:
        """計算最常見距離"""
        if not distances:
            return 48.0

        groups = defaultdict(list)
        for d in distances:
            groups[round(d / 2) * 2].append(d)

        if not groups:
            return 48.0

        return float(np.mean(max(groups.values(), key=len)))

    def apply_linear_regression_correction(
        self,
        points: np.ndarray,
        min_x: float,
        min_y: float,
        offset_x: float,
        offset_y: float,
    ) -> tuple[float, float, float, float]:
        """應用線性回歸校正"""
        # 建立有效點集合
        valid_points = []
        for x, y in points:
            col = round((x - min_x) / offset_x)
            row = round((y - min_y) / offset_y)
            if 0 <= col < 25 and 0 <= row < 10:
                valid_points.append((x, y, col, row))

        if len(valid_points) < 3:
            return min_x, min_y, offset_x, offset_y

        # X方向校正
        cols = [p[2] for p in valid_points]
        xs = [p[0] for p in valid_points]
        if len(set(cols)) > 1:
            coeffs = np.polyfit(cols, xs, 1)
            min_x, offset_x = coeffs[1], abs(coeffs[0])

        # Y方向校正
        rows = [p[3] for p in valid_points]
        ys = [p[1] for p in valid_points]
        if len(set(rows)) > 1:
            coeffs = np.polyfit(rows, ys, 1)
            min_y, offset_y = coeffs[1], abs(coeffs[0])

        return min_x, min_y, offset_x, offset_y

    def infer_board_parameters(
        self,
        grid_centers: list[tuple[int, int]],
    ) -> tuple[tuple[float, float], float, float, float]:
        """推理盤面參數"""
        if len(grid_centers) < 10:
            raise ValueError("方格數量不足，無法推理盤面")

        points = np.array(grid_centers)
        min_x, min_y = float(np.min(points[:, 0])), float(np.min(points[:, 1]))

        # 計算距離
        distances_x, distances_y = self.calculate_grid_distances(points)

        # 取得常見距離
        offset_x = self.get_common_distance(distances_x)
        offset_y = self.get_common_distance(distances_y)

        # 線性回歸校正
        min_x, min_y, offset_x, offset_y = self.apply_linear_regression_correction(
            points, min_x, min_y, offset_x, offset_y
        )

        return (min_x, min_y), min(offset_x, offset_y) * 0.9, offset_x, offset_y

    def validate_grid_coverage(
        self,
        grid_centers: list[tuple[int, int]],
        top_left: tuple[float, float],
        offset_x: float,
        offset_y: float,
    ) -> dict:
        """驗證方格覆蓋率"""
        rows, cols = self.grid_size
        total_expected = rows * cols

        grid_map = {}
        for x, y in grid_centers:
            col = round((x - top_left[0]) / offset_x)
            row = round((y - top_left[1]) / offset_y)
            if 0 <= col < cols and 0 <= row < rows:
                grid_map[(row, col)] = (x, y)

        found_grids = len(grid_map)
        coverage_ratio = found_grids / total_expected

        # 計算行列覆蓋率
        row_coverage = [
            sum(1 for c in range(cols) if (r, c) in grid_map) for r in range(rows)
        ]
        col_coverage = [
            sum(1 for r in range(rows) if (r, c) in grid_map) for c in range(cols)
        ]

        return {
            "found_grids": found_grids,
            "total_expected": total_expected,
            "coverage_ratio": coverage_ratio,
            "row_coverage": row_coverage,
            "col_coverage": col_coverage,
        }

    def validate_10x25_grid(
        self,
        grid_centers: list[tuple[int, int]],
        top_left: tuple[float, float],
        offset_x: float,
        offset_y: float,
    ) -> tuple[bool, dict]:
        """驗證 10×25 方格盤面"""
        validation_data = self.validate_grid_coverage(
            grid_centers, top_left, offset_x, offset_y
        )

        # 驗證條件
        coverage_ratio = validation_data["coverage_ratio"]
        row_coverage = validation_data["row_coverage"]
        col_coverage = validation_data["col_coverage"]

        is_valid = (
            coverage_ratio >= 0.7
            and min(row_coverage) >= 12
            and min(col_coverage) >= 4
            and sum(1 for x in row_coverage if x == 0) <= 2
            and sum(1 for x in col_coverage if x == 0) <= 5
        )

        validation_data["is_valid_10x25"] = is_valid
        return is_valid, validation_data

    def detect_board_from_image(
        self, image: np.ndarray, enable_validation: bool = True
    ) -> tuple[tuple[float, float], float, float, float]:
        """主要偵測函式"""
        try:
            grid_centers = self.detect_colored_grids_auto(image)
            top_left, grid_size, offset_x, offset_y = self.infer_board_parameters(
                grid_centers
            )

            if enable_validation:
                is_valid, validation_data = self.validate_10x25_grid(
                    grid_centers, top_left, offset_x, offset_y
                )
                logger.debug(
                    f"盤面驗證結果: 覆蓋率 {validation_data['coverage_ratio']:.2%}, 有效: {is_valid}"
                )

                if not is_valid:
                    logger.warning("偵測結果不符合 10×25 方格盤面要求，但繼續執行")

            return top_left, grid_size, offset_x, offset_y

        except Exception as e:
            logger.error(f"方格偵測失敗: {e}")
            raise

    def detect_board_from_screen_image(
        self, screen_image: np.ndarray, enable_validation: bool = True
    ) -> dict:
        """從螢幕截圖偵測盤面資訊"""
        import time

        start_time = time.perf_counter()

        try:
            # 將BGR轉換為RGB（因為mss截圖是BGR格式）
            if len(screen_image.shape) == 3 and screen_image.shape[2] == 3:
                # 假設輸入是BGR格式，轉為RGB
                image_rgb = cv2.cvtColor(screen_image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = screen_image

            top_left, grid_size, offset_x, offset_y = self.detect_board_from_image(
                image_rgb, enable_validation
            )

            execution_time = (time.perf_counter() - start_time) * 1000

            result = {
                "success": True,
                "top_left_corner": top_left,
                "grid_size": grid_size,
                "offset_x": offset_x,
                "offset_y": offset_y,
                "execution_time_ms": execution_time,
                "message": f"方格偵測成功，耗時 {execution_time:.1f} 毫秒",
            }

            logger.info(f"✅ 盤面偵測完成，耗時 {execution_time:.1f} 毫秒")
            logger.debug(
                f"左上角: ({top_left[0]:.2f}, {top_left[1]:.2f}), 方格: {grid_size:.2f}px, 偏移: X={offset_x:.2f}, Y={offset_y:.2f}"
            )

            return result

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            error_msg = f"方格偵測失敗: {e}"
            logger.error(error_msg)

            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time,
                "message": error_msg,
            }
