# 故障排除指南

> 🔧 Make10 專案常見問題的診斷方法與解決方案

## 🚨 快速診斷

### 系統健康檢查
```bash
# 1. 基礎環境檢查
python --version                    # 需要 Python 3.12+
uv --version                       # 確認 UV 套件管理器

# 2. 關鍵套件檢查
uv run python -c "import cv2, numpy, tensorflow; print('✅ 核心套件正常')"

# 3. 專案模組檢查
uv run python -c "from src.automation.screen_utils import capture_screen; print('✅ 螢幕工具正常')"
uv run python -c "from config.settings import cfg; print('✅ 配置系統正常')"

# 4. 測試執行檢查
uv run pytest tests/ -x --tb=short
```

### 快速修復指令
```bash
# 重新安裝環境
uv sync --dev

# 清理快取重新安裝
uv pip uninstall -r uv.lock
uv sync --dev

# 檢查並修復權限
# (需要管理員權限)
```

## 🛠️ 安裝問題

### UV 套件管理器問題

#### 問題：UV 安裝失敗
```
錯誤：'uv' 不是內部或外部命令
```

**解決方案：**
```powershell
# 方法 1：重新安裝 UV
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 方法 2：手動下載安裝
# 從 https://github.com/astral-sh/uv/releases 下載

# 方法 3：使用 pip 替代
pip install -e .[dev]
```

#### 問題：UV 權限錯誤
```
錯誤：拒絕存取
```

**解決方案：**
```powershell
# 設定執行政策
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 以管理員身分執行 PowerShell
Start-Process powershell -Verb runAs
```

### Python 套件問題

#### 問題：OpenCV 安裝失敗
```
錯誤：Failed building wheel for opencv-python
```

**解決方案：**
```bash
# 方法 1：使用預編譯版本
uv pip install opencv-python-headless==4.12.0.88

# 方法 2：更新 pip 和 setuptools
uv pip install --upgrade pip setuptools wheel

# 方法 3：使用 conda (替代方案)
conda install opencv
```

#### 問題：TensorFlow 相容性問題
```
錯誤：Your CPU supports instructions that this TensorFlow binary was not compiled to use
```

**解決方案：**
```bash
# 設定環境變數忽略警告
set TF_CPP_MIN_LOG_LEVEL=2

# 或安裝 CPU 最佳化版本
uv pip install tensorflow-cpu==2.19.0
```

## 🚀 系統執行問題

### 系統啟動失敗

#### 問題：模組匯入錯誤
```
ModuleNotFoundError: No module named 'src'
```

**診斷：**
```bash
# 檢查目前工作目錄
pwd

# 檢查 Python 路徑
uv run python -c "import sys; print('\n'.join(sys.path))"
```

**解決方案：**
```bash
# 確保在專案根目錄執行
cd make10-solver
uv run run_system.py

# 或使用絕對路徑
uv run python c:\Users\lingz\Workflow\make10\run_system.py
```

#### 問題：配置檔案載入失敗
```
FileNotFoundError: 配置檔案不存在
```

**解決方案：**
```bash
# 檢查配置檔案
ls config/settings.py
ls config/constants.py

# 重新建立配置 (如果遺失)
mkdir -p config
# 從 Git 恢復檔案
git checkout config/
```

### 鍵盤監聽問題

#### 問題：熱鍵無效
```
警告：鍵盤監聽器啟動失敗
```

**診斷：**
```python
# 測試鍵盤權限
uv run python -c "from pynput import keyboard; print('鍵盤模組正常')"
```

**解決方案：**
```powershell
# Windows：確認應用程式權限
# 設定 > 隱私權 > 其他裝置 > 允許存取鍵盤

# 或以管理員身分執行
Start-Process powershell -Verb runAs
cd make10-solver
uv run run_system.py
```

### 螢幕擷取問題

#### 問題：螢幕擷取失敗
```
錯誤：無法存取螢幕
```

**診斷：**
```python
# 測試螢幕擷取
uv run python -c "
from src.automation.screen_utils import capture_screen
result = capture_screen()
print(f'擷取結果: {result is not None}')
"
```

**解決方案：**
```powershell
# Windows：螢幕錄製權限
# 設定 > 隱私權 > 螢幕錄製 > 允許應用程式

# 檢查防毒軟體設定
# 將 python.exe 加入白名單
```

## 🧪 測試問題

### 測試執行失敗

#### 問題：測試無法找到模組
```
ImportError: attempted relative import with no known parent package
```

**解決方案：**
```bash
# 確保使用正確的測試指令
uv run pytest tests/ -v

# 而非直接執行
python tests/test_screen_utils.py  # ❌ 錯誤
```

#### 問題：Mock 測試失敗
```
AttributeError: Mock object has no attribute 'xxx'
```

**診斷：**
```bash
# 檢查特定測試
uv run pytest tests/test_screen_utils.py::TestScreenUtils::test_switch_screen_success -v -s
```

**解決方案：**
```python
# 檢查 Mock 設定
# 確保 patch 路徑正確
@patch('src.automation.screen_utils.keyboard')  # ✅ 正確
# 而非
@patch('pynput.keyboard')  # ❌ 錯誤
```

## � 效能問題

### 記憶體使用過高

#### 診斷：
```python
# 記憶體監控
uv run python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'記憶體使用: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

**解決方案：**
```python
# 1. 及時釋放大物件
large_array = np.zeros((1000, 1000))
# 使用後立即釋放
del large_array

# 2. 使用生成器代替列表
def process_data():
    for item in large_dataset:
        yield process_item(item)

# 3. 限制圖片快取大小
```

### CPU 使用率過高

**診斷：**
```bash
# Windows 工作管理員
tasklist | findstr python

# 或使用 psutil
uv run python -c "
import psutil
for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
    if 'python' in proc.info['name']:
        print(proc.info)
"
```

**解決方案：**
```python
# 1. 增加適當延遲
time.sleep(0.1)  # 主迴圈中

# 2. 最佳化圖片處理
# 使用更小的圖片尺寸
# 降低處理頻率

# 3. 使用多執行緒
from concurrent.futures import ThreadPoolExecutor
```

## 📝 日誌問題

### 日誌檔案過大

**檢查：**
```bash
# 檢查日誌檔案大小
ls -lh logs/

# 檢查日誌輪轉設定
grep -n "rotation" run_system.py
```

**解決方案：**
```python
# 調整日誌設定 (run_system.py)
logger.add(
    "logs/make10_system.log",
    rotation="1 day",      # 每日輪轉
    retention="7 days",    # 保留 7 天
    compression="zip"      # 壓縮舊檔案
)
```

### 日誌權限問題

**錯誤：**
```
PermissionError: [Errno 13] Permission denied: 'logs/make10_system.log'
```

**解決方案：**
```bash
# 檢查並修正權限
mkdir -p logs
chmod 755 logs/

# Windows
icacls logs /grant Users:(F)
```

## 🔍 除錯工具

### 除錯模式啟用
```python
# config/settings.py
cfg.SYSTEM.debug_mode = True

# 或環境變數
set DEBUG=1
uv run run_system.py
```

### 詳細日誌輸出
```python
# 暫時提高日誌等級
logger.add(sys.stderr, level="DEBUG")

# 或使用 print 除錯
print(f"變數值: {variable}")
```

### 互動式除錯
```python
# 在程式碼中插入除錯點
import pdb; pdb.set_trace()

# 或使用 IPython
import IPython; IPython.embed()
```

## 📞 取得協助

### 自助診斷腳本
```bash
# 建立診斷腳本 scripts/diagnose.ps1
Write-Host "🔍 開始系統診斷..." -ForegroundColor Green

# Python 環境
Write-Host "Python 版本:" -ForegroundColor Yellow
python --version

# UV 狀態
Write-Host "UV 版本:" -ForegroundColor Yellow  
uv --version

# 套件狀態
Write-Host "關鍵套件檢查:" -ForegroundColor Yellow
uv run python -c "
try:
    import cv2, numpy, tensorflow, pynput, loguru
    print('✅ 所有關鍵套件正常')
except ImportError as e:
    print(f'❌ 套件問題: {e}')
"

# 測試執行
Write-Host "快速測試:" -ForegroundColor Yellow
uv run pytest tests/test_config_settings.py::TestConfigurationObject::test_config_loading -v
```

### 問題回報格式
```markdown
## 問題描述
簡述遇到的問題

## 環境資訊
- OS: Windows 11
- Python: 3.12.x
- UV: x.x.x

## 重現步驟
1. 執行 uv run run_system.py
2. 出現錯誤訊息

## 錯誤訊息
```
完整的錯誤訊息
```

## 嘗試過的解決方案
- [ ] 重新安裝套件
- [ ] 檢查權限設定
```

### 相關資源
- 🐙 [GitHub Issues](https://github.com/lingzinc/make10-solver/issues)
- 📚 [專案文件](../README.md)
- 🛠️ [安裝指南](./installation.md)
- 🧪 [測試指南](./testing-guide.md)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 方案 2: 手動下載安裝
$url = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
Invoke-WebRequest -Uri $url -OutFile "uv.zip"
Expand-Archive -Path "uv.zip" -DestinationPath "C:\uv"
$env:PATH += ";C:\uv"

# 方案 3: 使用傳統 pip
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 問題 2: TensorFlow 安裝問題

#### 症狀
```
ImportError: DLL load failed while importing _pywrap_tensorflow_internal
```

#### 診斷步驟
```bash
# 檢查 Python 架構
uv run python -c "import platform; print(platform.architecture())"

# 檢查可用的 TensorFlow 版本
uv show tensorflow

# 檢查 Visual C++ Redistributable
```

#### 解決方案
```bash
# 方案 1: 重新安裝 TensorFlow
uv remove tensorflow
uv add "tensorflow==2.13.0"

# 方案 2: 使用 CPU 版本
uv add tensorflow-cpu

# 方案 3: 安裝 Visual C++ Redistributable
# 下載並安裝 Microsoft Visual C++ 2019-2022 Redistributable
```

### 問題 3: OpenCV 匯入錯誤

#### 症狀
```
ImportError: No module named 'cv2'
```

#### 解決方案
```bash
# 方案 1: 重新安裝 OpenCV
uv remove opencv-python
uv add opencv-python-headless

# 方案 2: 檢查衝突套件
uv list | grep opencv
uv remove opencv-contrib-python  # 如果存在衝突套件

# 方案 3: 手動安裝
pip install opencv-python==4.8.0.74
```

## 🧠 AI 模型相關問題

### 問題 4: 模型載入失敗

#### 症狀
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/models/model.keras'
```

#### 診斷步驟
```bash
# 檢查模型檔案
ls -la data/models/
file data/models/model.keras  # 檢查檔案類型

# 檢查檔案權限
ls -l data/models/model.keras
```

#### 解決方案
```bash
# 方案 1: 重新訓練模型
uv run python run_training.py --new

# 方案 2: 使用 MNIST 預訓練模型
uv run python run_training.py --mnist

# 方案 3: 檢查路徑設定
uv run python -c "
from config.settings import Settings
s = Settings()
print(f'模型路徑: {s.model_path}')
print(f'路徑存在: {s.model_path.exists()}')
"
```

### 問題 5: AI 預測準確率低

#### 症狀
```
AI 識別數字錯誤率高，信心度普遍低於 0.5
```

#### 診斷步驟
```python
# 檢查模型資訊
uv run python -c "
import tensorflow as tf
model = tf.keras.models.load_model('data/models/model.keras')
print(model.summary())
print(f'輸入形狀: {model.input_shape}')
print(f'輸出形狀: {model.output_shape}')
"

# 檢查訓練資料品質
uv run python -c "
import pandas as pd
labels = pd.read_csv('data/training/labels.csv')
print(labels['label'].value_counts())
print(f'總樣本數: {len(labels)}')
"
```

#### 解決方案
```bash
# 方案 1: 增加訓練資料
uv run python run_training.py --label  # 手動標註更多資料

# 方案 2: 調整模型架構
# 編輯 run_training.py 中的模型結構

# 方案 3: 使用資料增強
uv run python run_training.py --augment

# 方案 4: 重新評估模型
uv run python -c "
from src.ai.predictor import Predictor
predictor = Predictor()
# 在已知資料上測試準確率
"
```

## 🖥️ 螢幕擷取問題

### 問題 6: 螢幕擷取失敗

#### 症狀
```
ScreenCaptureError: Failed to capture screen
```

#### 診斷步驟
```python
# 測試 MSS 功能
uv run python -c "
import mss
with mss.mss() as sct:
    print(f'可用螢幕數: {len(sct.monitors)}')
    for i, monitor in enumerate(sct.monitors):
        print(f'螢幕 {i}: {monitor}')
"

# 檢查權限
# Windows: 確認程式具有螢幕錄製權限
```

#### 解決方案
```bash
# 方案 1: 以管理員身分執行
# 右鍵點擊 PowerShell -> 以系統管理員身分執行

# 方案 2: 檢查防毒軟體設定
# 將專案目錄加入防毒軟體白名單

# 方案 3: 使用替代螢幕擷取方法
uv run python -c "
import PIL.ImageGrab as ImageGrab
screenshot = ImageGrab.grab()
print(f'螢幕尺寸: {screenshot.size}')
"
```

### 問題 7: 多螢幕支援問題

#### 症狀
```
遊戲在副螢幕但系統只能偵測主螢幕
```

#### 解決方案
```python
# 檢查所有螢幕
uv run python -c "
import mss
with mss.mss() as sct:
    for i, monitor in enumerate(sct.monitors[1:], 1):
        screenshot = sct.grab(monitor)
        print(f'螢幕 {i}: {monitor}')
        # 儲存截圖檢查
        mss.tools.to_png(screenshot.rgb, screenshot.size, 
                         output=f'screen_{i}.png')
"

# 手動指定螢幕
uv run python -c "
from src.automation.screen_utils import ScreenUtils
screen_utils = ScreenUtils()
screen_utils.switch_to_screen(2)  # 切換到第二個螢幕
"
```

## 🎮 遊戲整合問題

### 問題 8: 無法偵測遊戲視窗

#### 症狀
```
GameNotFoundError: Unable to locate game window
```

#### 診斷步驟
```python
# 檢查重置按鈕模板
uv run python -c "
import cv2
import numpy as np
from pathlib import Path

template_path = Path('data/assets/templates/reset_button_b.png')
if template_path.exists():
    template = cv2.imread(str(template_path))
    print(f'模板尺寸: {template.shape}')
else:
    print('重置按鈕模板不存在')
"

# 測試模板匹配
uv run python -c "
from src.automation.screen_utils import ScreenUtils
screen_utils = ScreenUtils()
screenshot = screen_utils.capture_screen()
# 手動檢查截圖中是否包含重置按鈕
"
```

#### 解決方案
```bash
# 方案 1: 更新重置按鈕模板
# 手動截取當前遊戲的重置按鈕圖像
# 儲存為 data/assets/templates/reset_button_custom.png

# 方案 2: 調整匹配閾值
uv run python -c "
from config.settings import Settings
settings = Settings()
settings.template_match_threshold = 0.7  # 降低匹配要求
"

# 方案 3: 使用 OCR 輔助偵測
uv add pytesseract
# 實作 OCR 輔助的遊戲偵測
```

### 問題 9: 滑鼠控制失效

#### 症狀
```
滑鼠移動正常但無法在遊戲中觸發點擊
```

#### 診斷步驟
```python
# 測試滑鼠基本功能
uv run python -c "
from pynput import mouse
import time

def on_click(x, y, button, pressed):
    print(f'滑鼠點擊: {x}, {y}, {button}, {pressed}')
    return False  # 停止監聽

# 啟動監聽器測試
listener = mouse.Listener(on_click=on_click)
listener.start()
print('請點擊滑鼠測試...')
listener.join(timeout=5)
"
```

#### 解決方案
```python
# 方案 1: 調整點擊延遲
from src.automation.mouse_controller import MouseController
controller = MouseController()
controller.click_delay = 0.1  # 增加點擊間隔

# 方案 2: 使用不同的點擊方法
controller.use_win32_api = True  # 使用 Windows API

# 方案 3: 檢查遊戲焦點
import win32gui
def bring_game_to_front():
    hwnd = win32gui.FindWindow(None, "Make10")
    if hwnd:
        win32gui.SetForegroundWindow(hwnd)
```

## 📊 效能問題

### 問題 10: 記憶體使用過高

#### 症狀
```
系統記憶體使用持續增長，最終導致程式崩潰
```

#### 診斷步驟
```python
# 記憶體使用監控
uv run python -c "
import psutil
import time

process = psutil.Process()
for i in range(10):
    memory_info = process.memory_info()
    print(f'時間 {i}: RSS={memory_info.rss/1024/1024:.1f}MB, VMS={memory_info.vms/1024/1024:.1f}MB')
    time.sleep(1)
"

# 檢查 GPU 記憶體 (如果有 GPU)
uv run python -c "
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f'GPU: {gpu}')
        # 檢查記憶體使用
"
```

#### 解決方案
```python
# 方案 1: 啟用 TensorFlow 記憶體增長
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 方案 2: 實作記憶體清理
import gc
def cleanup_memory():
    gc.collect()
    tf.keras.backend.clear_session()

# 方案 3: 減少批次大小
# 在 config/settings.py 中調整
BATCH_SIZE = 32  # 降低批次大小
```

### 問題 11: 執行速度過慢

#### 症狀
```
完整遊戲週期超過 10 秒，系統回應緩慢
```

#### 診斷步驟
```python
# 效能分析
import time
import cProfile

def profile_game_cycle():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 執行遊戲週期
    run_game_cycle()
    
    profiler.disable()
    profiler.print_stats(sort='cumulative')

# 分段計時
times = {}
start = time.time()
# ... 執行各階段 ...
times['scan'] = time.time() - start
```

#### 解決方案
```bash
# 方案 1: 啟用多執行緒
export OMP_NUM_THREADS=4
export TF_NUM_INTEROP_THREADS=4
export TF_NUM_INTRAOP_THREADS=4

# 方案 2: 最佳化圖像處理
# 減少圖像解析度
# 使用更快的演算法

# 方案 3: 快取預測結果
# 實作預測結果快取機制
```

## 🔧 日誌與除錯

### 啟用詳細日誌
```python
# 設定日誌等級
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/debug.log'),
        logging.StreamHandler()
    ]
)

# 啟用 TensorFlow 詳細日誌
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
```

### 除錯工具
```bash
# 使用 Python 除錯器
uv run python -m pdb run_system.py

# 使用 IPython 進行互動式除錯
uv add ipython
uv run ipython
```

### 效能監控
```python
# 實時效能監控
import threading
import time

def monitor_performance():
    while True:
        # 監控 CPU、記憶體、GPU 使用率
        time.sleep(5)

monitor_thread = threading.Thread(target=monitor_performance, daemon=True)
monitor_thread.start()
```

## 📞 獲取協助

### 收集診斷資訊
```bash
# 產生診斷報告
uv run python -c "
import sys
import platform
import tensorflow as tf
import cv2
import numpy as np

print('=== 系統資訊 ===')
print(f'作業系統: {platform.system()} {platform.release()}')
print(f'Python 版本: {sys.version}')
print(f'TensorFlow 版本: {tf.__version__}')
print(f'OpenCV 版本: {cv2.__version__}')
print(f'NumPy 版本: {np.__version__}')

print('\n=== GPU 資訊 ===')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPU 數量: {len(gpus)}')
for gpu in gpus:
    print(f'GPU: {gpu}')
"
```

### 問題回報格式
```markdown
## 問題描述
[簡述問題現象]

## 復現步驟
1. [步驟一]
2. [步驟二]
3. [問題出現]

## 預期行為
[應該發生什麼]

## 實際行為
[實際發生什麼]

## 環境資訊
- 作業系統: [Windows 版本]
- Python 版本: [版本號]
- 專案版本: [Git commit hash]

## 錯誤訊息
```
[貼上完整錯誤訊息]
```

## 診斷結果
[貼上診斷指令的輸出]
```

如果遇到本指南未涵蓋的問題，請收集相關診斷資訊並透過 GitHub Issues 回報問題。
