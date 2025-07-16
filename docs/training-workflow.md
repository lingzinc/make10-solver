# 模型訓練工作流程

🎓 本指南詳細說明 Make10 專案中 AI 模型的訓練工作流程、資料管理與模型生命週期。

## 🔄 訓練工作流程概覽

### 完整訓練管道
```
資料收集 → 資料標註 → 預處理 → 模型訓練 → 驗證評估 → 模型部署 → 監控反饋
    ↓           ↓           ↓           ↓           ↓           ↓           ↓
📷 擷取        🏷️ 人工      🔧 增強      🧠 CNN      📊 測試      🚀 上線      📈 調整
Cell 圖像      標註數字     正規化       訓練        準確率       模型        效能
```

## 📊 資料管理工作流程

### 1. 資料收集階段

#### 自動化資料收集
```python
def automated_data_collection(num_games=10, cells_per_game=250):
    """自動化遊戲資料收集"""
    
    collected_data = []
    
    for game_idx in range(num_games):
        print(f"收集第 {game_idx + 1} 場遊戲資料...")
        
        # 啟動遊戲並等待穩定
        start_new_game()
        time.sleep(2)
        
        # 擷取完整盤面
        board_image = capture_game_board()
        
        # 分割成個別 cell
        cell_images = extract_cell_images(board_image)
        
        # 儲存每個 cell
        for cell_idx, cell_img in enumerate(cell_images):
            filename = f"game_{game_idx:03d}_cell_{cell_idx:03d}.png"
            filepath = save_cell_image(cell_img, filename)
            
            collected_data.append({
                'filename': filename,
                'filepath': filepath,
                'game_id': game_idx,
                'cell_id': cell_idx,
                'timestamp': datetime.now()
            })
    
    # 儲存收集資訊
    save_collection_metadata(collected_data)
    print(f"總共收集 {len(collected_data)} 個 cell 圖像")
    
    return collected_data

def extract_cell_images(board_image, grid_size=(5, 5)):
    """從盤面圖像中提取個別 cell"""
    
    # 檢測網格線
    grid_lines = detect_grid_lines(board_image)
    
    # 計算每個 cell 的座標
    cell_coordinates = calculate_cell_coordinates(grid_lines, grid_size)
    
    cell_images = []
    for i, (x, y, w, h) in enumerate(cell_coordinates):
        # 提取 cell 區域
        cell = board_image[y:y+h, x:x+w]
        
        # 調整大小到標準尺寸
        cell_resized = cv2.resize(cell, (45, 45))
        
        cell_images.append(cell_resized)
    
    return cell_images
```

### 2. 資料標註階段

#### 互動式標註工具
```python
class InteractiveLabelingTool:
    """互動式資料標註工具"""
    
    def __init__(self, data_directory="data/training/images"):
        self.data_dir = Path(data_directory)
        self.labels_file = Path("data/training/labels.csv")
        self.current_labels = self.load_existing_labels()
        
    def start_labeling_session(self):
        """開始標註會話"""
        
        unlabeled_images = self.get_unlabeled_images()
        print(f"找到 {len(unlabeled_images)} 個未標註的圖像")
        
        for img_path in unlabeled_images:
            try:
                label = self.label_single_image(img_path)
                if label is not None:
                    self.save_label(img_path, label)
                    
            except KeyboardInterrupt:
                print("\n標註會話被使用者中斷")
                break
            except Exception as e:
                print(f"標註圖像 {img_path} 時發生錯誤: {e}")
                continue
        
        print("標註會話完成")
    
    def label_single_image(self, img_path):
        """標註單一圖像"""
        
        # 載入並顯示圖像
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"無法載入圖像: {img_path}")
            return None
        
        # 放大顯示以便識別
        display_image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow(f"標註: {img_path.name}", display_image)
        cv2.moveWindow(f"標註: {img_path.name}", 100, 100)
        
        print(f"\n標註圖像: {img_path.name}")
        print("請輸入數字 (0-9), 或按 's' 跳過, 'q' 退出:")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            # 數字鍵 (0-9)
            if ord('0') <= key <= ord('9'):
                label = key - ord('0')
                cv2.destroyAllWindows()
                return label
            
            # 跳過
            elif key == ord('s'):
                print("跳過此圖像")
                cv2.destroyAllWindows()
                return None
            
            # 退出
            elif key == ord('q'):
                cv2.destroyAllWindows()
                raise KeyboardInterrupt()
            
            # 無效輸入
            else:
                print("無效輸入，請輸入 0-9 或 's'/'q'")
    
    def get_unlabeled_images(self):
        """取得未標註的圖像列表"""
        
        all_images = list(self.data_dir.glob("*.png"))
        labeled_images = set(self.current_labels.keys())
        
        unlabeled = [img for img in all_images 
                    if img.name not in labeled_images]
        
        return sorted(unlabeled)
    
    def save_label(self, img_path, label):
        """儲存標籤"""
        
        self.current_labels[img_path.name] = label
        
        # 更新 CSV 檔案
        with open(self.labels_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'label'])
            
            for filename, label in self.current_labels.items():
                writer.writerow([filename, label])
        
        print(f"已儲存: {img_path.name} → {label}")
    
    def load_existing_labels(self):
        """載入現有標籤"""
        
        labels = {}
        
        if self.labels_file.exists():
            with open(self.labels_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    labels[row['filename']] = int(row['label'])
        
        return labels
```

#### 標註品質控制
```python
class LabelQualityController:
    """標註品質控制器"""
    
    def __init__(self, labels_file="data/training/labels.csv"):
        self.labels_file = Path(labels_file)
        self.labels_df = pd.read_csv(labels_file)
    
    def analyze_label_distribution(self):
        """分析標籤分佈"""
        
        distribution = self.labels_df['label'].value_counts().sort_index()
        
        print("=== 標籤分佈分析 ===")
        for label, count in distribution.items():
            percentage = count / len(self.labels_df) * 100
            print(f"數字 {label}: {count} 個樣本 ({percentage:.1f}%)")
        
        # 檢查資料平衡性
        min_count = distribution.min()
        max_count = distribution.max()
        imbalance_ratio = max_count / min_count
        
        if imbalance_ratio > 3:
            print(f"\n⚠️  資料不平衡警告: 最大/最小比例 = {imbalance_ratio:.1f}")
            print("建議增加樣本數量較少的類別")
        
        return distribution
    
    def find_duplicate_labels(self):
        """尋找重複標籤"""
        
        # 載入圖像並計算雜湊
        image_hashes = {}
        duplicates = []
        
        for _, row in self.labels_df.iterrows():
            img_path = Path("data/training/images") / row['filename']
            
            if img_path.exists():
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                img_hash = hashlib.md5(image.tobytes()).hexdigest()
                
                if img_hash in image_hashes:
                    duplicates.append({
                        'hash': img_hash,
                        'files': [image_hashes[img_hash], row['filename']],
                        'labels': [self.get_label_for_file(image_hashes[img_hash]), row['label']]
                    })
                else:
                    image_hashes[img_hash] = row['filename']
        
        if duplicates:
            print(f"發現 {len(duplicates)} 組重複圖像")
            for dup in duplicates:
                print(f"檔案: {dup['files']}, 標籤: {dup['labels']}")
        
        return duplicates
    
    def validate_labels_with_model(self, model_path="data/models/model.keras"):
        """使用現有模型驗證標籤"""
        
        if not Path(model_path).exists():
            print("模型檔案不存在，跳過驗證")
            return
        
        model = tf.keras.models.load_model(model_path)
        inconsistencies = []
        
        for _, row in self.labels_df.iterrows():
            img_path = Path("data/training/images") / row['filename']
            
            if img_path.exists():
                # 載入並預處理圖像
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                processed = preprocess_for_prediction(image)
                
                # 模型預測
                prediction = model.predict(processed, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)
                
                # 檢查不一致性
                if predicted_class != row['label'] and confidence > 0.8:
                    inconsistencies.append({
                        'filename': row['filename'],
                        'human_label': row['label'],
                        'model_prediction': predicted_class,
                        'confidence': confidence
                    })
        
        if inconsistencies:
            print(f"發現 {len(inconsistencies)} 個可能的標註錯誤")
            for inc in inconsistencies[:10]:  # 只顯示前 10 個
                print(f"{inc['filename']}: 人工={inc['human_label']}, "
                      f"模型={inc['model_prediction']} (信心度={inc['confidence']:.3f})")
        
        return inconsistencies
```

### 3. 資料預處理階段

#### 進階資料增強
```python
class AdvancedDataAugmentation:
    """進階資料增強器"""
    
    def __init__(self, augmentation_factor=3):
        self.aug_factor = augmentation_factor
        
    def augment_dataset(self, images, labels):
        """對整個資料集進行增強"""
        
        augmented_images = []
        augmented_labels = []
        
        for img, label in zip(images, labels):
            # 原始圖像
            augmented_images.append(img)
            augmented_labels.append(label)
            
            # 生成增強變體
            for _ in range(self.aug_factor):
                aug_img = self.apply_random_augmentation(img)
                augmented_images.append(aug_img)
                augmented_labels.append(label)
        
        return np.array(augmented_images), np.array(augmented_labels)
    
    def apply_random_augmentation(self, image):
        """隨機應用增強技術"""
        
        aug_image = image.copy()
        
        # 隨機選擇增強技術
        augmentations = [
            self.rotate_image,
            self.scale_image,
            self.translate_image,
            self.add_noise,
            self.adjust_brightness,
            self.adjust_contrast
        ]
        
        # 隨機選擇 1-3 種增強技術
        num_augs = random.randint(1, 3)
        selected_augs = random.sample(augmentations, num_augs)
        
        for aug_func in selected_augs:
            aug_image = aug_func(aug_image)
        
        return aug_image
    
    def rotate_image(self, image, max_angle=15):
        """旋轉圖像"""
        angle = random.uniform(-max_angle, max_angle)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        return rotated
    
    def scale_image(self, image, scale_range=(0.8, 1.2)):
        """縮放圖像"""
        scale = random.uniform(*scale_range)
        height, width = image.shape[:2]
        
        new_height, new_width = int(height * scale), int(width * scale)
        scaled = cv2.resize(image, (new_width, new_height))
        
        # 裁剪或填充到原始大小
        if scale > 1.0:
            # 縮放後較大，需要裁剪
            start_y = (new_height - height) // 2
            start_x = (new_width - width) // 2
            scaled = scaled[start_y:start_y+height, start_x:start_x+width]
        else:
            # 縮放後較小，需要填充
            padded = np.zeros_like(image)
            start_y = (height - new_height) // 2
            start_x = (width - new_width) // 2
            padded[start_y:start_y+new_height, start_x:start_x+new_width] = scaled
            scaled = padded
        
        return scaled
    
    def add_noise(self, image, noise_factor=0.1):
        """添加高斯雜訊"""
        noise = np.random.normal(0, noise_factor * 255, image.shape)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return noisy
```

## 🧠 模型訓練階段

### 進階訓練管道
```python
class AdvancedTrainingPipeline:
    """進階訓練管道"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        
    def train_with_cross_validation(self, X, y, k_folds=5):
        """使用交叉驗證訓練"""
        
        from sklearn.model_selection import StratifiedKFold
        
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_scores = []
        best_model = None
        best_score = 0
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y.argmax(axis=1))):
            print(f"\n=== 訓練第 {fold + 1} 折 ===")
            
            # 分割資料
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # 建立新模型
            model = self.create_model()
            
            # 訓練模型
            history = model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                callbacks=self.create_callbacks(fold),
                verbose=1
            )
            
            # 評估模型
            val_score = model.evaluate(X_val_fold, y_val_fold, verbose=0)[1]
            cv_scores.append(val_score)
            
            print(f"第 {fold + 1} 折驗證準確率: {val_score:.4f}")
            
            # 保存最佳模型
            if val_score > best_score:
                best_score = val_score
                best_model = model
        
        print(f"\n交叉驗證結果:")
        print(f"平均準確率: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        print(f"最佳準確率: {best_score:.4f}")
        
        return best_model, cv_scores
    
    def train_with_early_stopping(self, X_train, y_train, X_val, y_val):
        """使用早停機制訓練"""
        
        model = self.create_model()
        
        # 早停回調
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # 學習率衰減
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # 模型檢查點
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/best_model_epoch_{epoch:02d}.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # 訓練模型
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['max_epochs'],
            batch_size=self.config['batch_size'],
            callbacks=[early_stopping, lr_scheduler, checkpoint],
            verbose=1
        )
        
        return model, history
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val):
        """超參數調整"""
        
        import itertools
        
        # 定義超參數搜尋空間
        param_grid = {
            'learning_rate': [0.001, 0.0005, 0.0001],
            'batch_size': [16, 32, 64],
            'dropout_rate': [0.3, 0.5, 0.7],
            'num_filters': [16, 32, 64]
        }
        
        best_params = None
        best_score = 0
        results = []
        
        # 網格搜尋
        param_combinations = list(itertools.product(*param_grid.values()))
        
        for i, params in enumerate(param_combinations):
            param_dict = dict(zip(param_grid.keys(), params))
            print(f"\n=== 測試參數組合 {i+1}/{len(param_combinations)} ===")
            print(f"參數: {param_dict}")
            
            try:
                # 使用當前參數建立和訓練模型
                model = self.create_model_with_params(param_dict)
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,  # 較少的 epochs 用於快速評估
                    batch_size=param_dict['batch_size'],
                    verbose=0
                )
                
                # 評估模型
                val_score = max(history.history['val_accuracy'])
                
                results.append({
                    'params': param_dict,
                    'score': val_score
                })
                
                print(f"驗證準確率: {val_score:.4f}")
                
                if val_score > best_score:
                    best_score = val_score
                    best_params = param_dict
                    
            except Exception as e:
                print(f"參數組合失敗: {e}")
                continue
        
        print(f"\n=== 超參數調整結果 ===")
        print(f"最佳參數: {best_params}")
        print(f"最佳分數: {best_score:.4f}")
        
        return best_params, results
```

### 模型訓練監控
```python
class TrainingMonitor:
    """訓練過程監控器"""
    
    def __init__(self, log_dir="logs/training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
    def create_tensorboard_callback(self, experiment_name):
        """建立 TensorBoard 回調"""
        
        log_path = self.log_dir / experiment_name
        
        return tf.keras.callbacks.TensorBoard(
            log_dir=str(log_path),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
    
    def create_custom_logging_callback(self):
        """建立自定義日誌回調"""
        
        class CustomLoggingCallback(tf.keras.callbacks.Callback):
            def __init__(self, monitor):
                super().__init__()
                self.monitor = monitor
                
            def on_epoch_end(self, epoch, logs=None):
                # 記錄詳細的訓練資訊
                self.monitor.log_epoch_metrics(epoch, logs)
                
                # 檢查異常情況
                if logs.get('loss', 0) > 10:
                    print("警告: 損失值過高，可能存在梯度爆炸")
                
                if logs.get('val_accuracy', 0) < 0.1:
                    print("警告: 驗證準確率過低，檢查資料或模型")
        
        return CustomLoggingCallback(self)
    
    def log_epoch_metrics(self, epoch, logs):
        """記錄每個 epoch 的指標"""
        
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'epoch': epoch,
            'metrics': logs
        }
        
        # 儲存到 JSON 日誌
        log_file = self.log_dir / "training_log.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def plot_training_history(self, history, save_path=None):
        """繪製訓練歷史"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 準確率
        axes[0, 0].plot(history.history['accuracy'], label='Training')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # 損失
        axes[0, 1].plot(history.history['loss'], label='Training')
        axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # 學習率 (如果有記錄)
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('LR')
        
        # 梯度範數 (如果有記錄)
        if 'gradient_norm' in history.history:
            axes[1, 1].plot(history.history['gradient_norm'])
            axes[1, 1].set_title('Gradient Norm')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Norm')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
```

## 🚀 模型部署與監控

### 自動化部署管道
```python
class ModelDeploymentPipeline:
    """模型部署管道"""
    
    def __init__(self):
        self.deployment_config = {
            'min_accuracy': 0.85,
            'max_model_size': 50 * 1024 * 1024,  # 50MB
            'max_inference_time': 100  # ms
        }
    
    def deploy_model(self, model_path, test_data):
        """部署模型到生產環境"""
        
        # 1. 載入模型
        model = tf.keras.models.load_model(model_path)
        
        # 2. 模型驗證
        validation_result = self.validate_model(model, test_data)
        
        if not validation_result['passed']:
            raise ValueError(f"模型驗證失敗: {validation_result['errors']}")
        
        # 3. 模型最佳化
        optimized_model_path = self.optimize_model(model)
        
        # 4. A/B 測試準備
        ab_test_config = self.prepare_ab_test(optimized_model_path)
        
        # 5. 逐步部署
        deployment_result = self.gradual_rollout(optimized_model_path, ab_test_config)
        
        return deployment_result
    
    def validate_model(self, model, test_data):
        """驗證模型是否符合部署標準"""
        
        X_test, y_test = test_data
        errors = []
        
        # 準確率檢查
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        if test_accuracy < self.deployment_config['min_accuracy']:
            errors.append(f"準確率不足: {test_accuracy:.3f} < {self.deployment_config['min_accuracy']}")
        
        # 模型大小檢查
        model_size = self.get_model_size(model)
        if model_size > self.deployment_config['max_model_size']:
            errors.append(f"模型過大: {model_size} bytes")
        
        # 推理時間檢查
        inference_time = self.measure_inference_time(model, X_test[:10])
        if inference_time > self.deployment_config['max_inference_time']:
            errors.append(f"推理時間過長: {inference_time:.1f}ms")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'metrics': {
                'accuracy': test_accuracy,
                'model_size': model_size,
                'inference_time': inference_time
            }
        }
    
    def gradual_rollout(self, model_path, ab_test_config):
        """逐步部署新模型"""
        
        rollout_stages = [0.05, 0.1, 0.25, 0.5, 1.0]  # 5%, 10%, 25%, 50%, 100%
        
        for stage_percent in rollout_stages:
            print(f"部署階段: {stage_percent*100:.0f}% 使用者")
            
            # 更新 A/B 測試比例
            self.update_ab_test_ratio(ab_test_config, stage_percent)
            
            # 監控一段時間
            time.sleep(300)  # 等待 5 分鐘
            
            # 檢查關鍵指標
            metrics = self.collect_deployment_metrics()
            
            if not self.check_deployment_health(metrics):
                print("部署健康檢查失敗，執行回滾")
                self.rollback_deployment()
                return {'status': 'failed', 'stage': stage_percent}
            
            print(f"階段 {stage_percent*100:.0f}% 部署成功")
        
        return {'status': 'success', 'deployed_at': datetime.now()}
```

### 生產環境監控
```python
class ProductionMonitor:
    """生產環境模型監控"""
    
    def __init__(self):
        self.metrics_buffer = []
        self.alert_thresholds = {
            'accuracy_drop': 0.05,
            'inference_time_spike': 2.0,
            'error_rate_spike': 0.1
        }
    
    def monitor_model_performance(self):
        """監控模型在生產環境的表現"""
        
        while True:
            try:
                # 收集最近的預測結果
                recent_predictions = self.get_recent_predictions()
                
                # 計算關鍵指標
                metrics = self.calculate_metrics(recent_predictions)
                
                # 檢查是否需要警報
                alerts = self.check_for_alerts(metrics)
                
                if alerts:
                    self.send_alerts(alerts)
                
                # 記錄指標
                self.log_metrics(metrics)
                
                time.sleep(60)  # 每分鐘檢查一次
                
            except Exception as e:
                logger.error(f"監控過程中發生錯誤: {e}")
                time.sleep(60)
    
    def calculate_metrics(self, predictions):
        """計算關鍵指標"""
        
        if not predictions:
            return {}
        
        metrics = {
            'timestamp': datetime.now(),
            'total_predictions': len(predictions),
            'avg_confidence': np.mean([p['confidence'] for p in predictions]),
            'low_confidence_rate': len([p for p in predictions if p['confidence'] < 0.8]) / len(predictions),
            'avg_inference_time': np.mean([p['inference_time'] for p in predictions]),
            'error_rate': len([p for p in predictions if p.get('error')]) / len(predictions)
        }
        
        return metrics
    
    def check_for_alerts(self, current_metrics):
        """檢查是否需要發送警報"""
        
        alerts = []
        
        # 比較歷史指標
        if len(self.metrics_buffer) > 0:
            baseline = self.calculate_baseline_metrics()
            
            # 準確率下降檢查
            if current_metrics.get('avg_confidence', 0) < baseline.get('avg_confidence', 1) - self.alert_thresholds['accuracy_drop']:
                alerts.append({
                    'type': 'accuracy_drop',
                    'message': f"模型信心度下降: {current_metrics['avg_confidence']:.3f} vs {baseline['avg_confidence']:.3f}",
                    'severity': 'high'
                })
            
            # 推理時間激增檢查
            if current_metrics.get('avg_inference_time', 0) > baseline.get('avg_inference_time', 0) * self.alert_thresholds['inference_time_spike']:
                alerts.append({
                    'type': 'inference_time_spike',
                    'message': f"推理時間激增: {current_metrics['avg_inference_time']:.1f}ms vs {baseline['avg_inference_time']:.1f}ms",
                    'severity': 'medium'
                })
            
            # 錯誤率激增檢查
            if current_metrics.get('error_rate', 0) > self.alert_thresholds['error_rate_spike']:
                alerts.append({
                    'type': 'error_rate_spike',
                    'message': f"錯誤率過高: {current_metrics['error_rate']:.3f}",
                    'severity': 'high'
                })
        
        return alerts
```

透過這個完整的訓練工作流程，Make10 專案能夠系統化地管理 AI 模型的整個生命週期，從資料收集到生產部署，確保模型品質和系統穩定性。
