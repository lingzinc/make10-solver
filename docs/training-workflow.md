# ResNet50 模型訓練工作流程

🎓 本指南詳細說明 Make10 專案中基於 ResNet50 的 AI 模型訓練工作流程、資料管理與模型生命週期。

## 🔄 ResNet50 訓練工作流程概覽

### 完整訓練管道
```
資料收集 → 資料標註 → 預處理 → 遷移學習 → 微調訓練 → 驗證評估 → 模型部署 → 監控反饋
    ↓           ↓           ↓           ↓           ↓           ↓           ↓           ↓
📷 擷取        🏷️ 人工      🔧 RGB      🧠 ImageNet  🎯 Fine     📊 測試      🚀 上線      📈 調整
Cell 圖像      標註數字     224x224     預訓練      Tuning      準確率       模型        效能
```

## 📊 ResNet50 專用資料管理

### 1. 資料收集階段 (針對 ResNet50 優化)

#### 高解析度資料收集
```python
def collect_high_resolution_data(num_games=10, target_size=(224, 224)):
    """為 ResNet50 收集高解析度資料"""
    
    collected_data = []
    
    for game_idx in range(num_games):
        print(f"收集第 {game_idx + 1} 場遊戲高解析度資料...")
        
        # 啟動遊戲並等待穩定
        start_new_game()
        time.sleep(2)
        
        # 擷取完整盤面 (高解析度)
        board_image = capture_game_board(scale_factor=2.0)  # 2倍解析度
        
        # 分割成個別 cell 並保持高解析度
        cell_images = extract_cell_images_hd(board_image, target_size)
        
        # 儲存每個 cell (保存為 RGB 格式)
        for cell_idx, cell_img in enumerate(cell_images):
            # 確保 RGB 格式
            if len(cell_img.shape) == 2:
                cell_img = cv2.cvtColor(cell_img, cv2.COLOR_GRAY2RGB)
            elif cell_img.shape[2] == 4:
                cell_img = cv2.cvtColor(cell_img, cv2.COLOR_BGRA2RGB)
            
            filename = f"resnet_game_{game_idx:03d}_cell_{cell_idx:03d}.png"
            filepath = save_cell_image_rgb(cell_img, filename)
            
            collected_data.append({
                'filename': filename,
                'filepath': filepath,
                'game_id': game_idx,
                'cell_id': cell_idx,
                'resolution': cell_img.shape,
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
                    best_score = val_score;
                    best_params = param_dict;
                    
            except Exception as e:
                print(f"參數組合失敗: {e}")
                continue
        
        print(f"\n=== 超參數調整結果 ===")
        print(f"最佳參數: {best_params}")
        print(f"最佳分數: {best_score:.4f}")
        
        return best_params, results
```

## 🏃‍♂️ ResNet50 訓練流程設計

### ResNet50 遷移學習管道
```python
# run_training.py - ResNet50 訓練入口
def main_resnet50_training_pipeline():
    """ResNet50 主要訓練流程"""
    
    print("🚀 開始 ResNet50 訓練流程...")
    
    # 1. 資料載入與預處理
    print("📊 載入訓練資料...")
    train_data, val_data, test_data = load_resnet50_training_data()
    
    # 2. 建立 ResNet50 模型
    print("🧠 建立 ResNet50 模型...")
    model = create_resnet50_digit_model(pretrained=True)
    
    # 3. 兩階段訓練策略
    print("🎯 開始兩階段訓練...")
    
    # 階段 1: 訓練分類頭
    print("⚡ 階段 1: 訓練分類頭 (凍結預訓練層)")
    model, history_1 = train_stage_1(model, train_data, val_data)
    
    # 階段 2: 微調整個網路
    print("🔧 階段 2: 微調整個網路")
    model, history_2 = train_stage_2(model, train_data, val_data)
    
    # 4. 模型評估
    print("📈 評估最終模型...")
    evaluation_results = evaluate_resnet50_model(model, test_data)
    
    # 5. 模型儲存
    print("💾 儲存訓練完成的模型...")
    save_final_model(model, evaluation_results)
    
    print("✅ ResNet50 訓練完成!")
    return model, history_1, history_2, evaluation_results

def train_stage_1(model, train_data, val_data):
    """階段 1: 凍結預訓練層，只訓練分類頭"""
    
    # 凍結 ResNet50 基礎模型
    model.layers[1].trainable = False  # base_model 凍結
    
    # 編譯模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_3_accuracy']
    )
    
    # 訓練配置
    callbacks_stage1 = [
        tf.keras.callbacks.ModelCheckpoint(
            'data/models/checkpoints/stage1_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='logs/stage1',
            histogram_freq=1
        )
    ]
    
    # 執行訓練
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=cfg.TRAINING.epochs_stage1,
        callbacks=callbacks_stage1,
        verbose=1
    )
    
    print(f"階段 1 完成 - 最佳驗證準確率: {max(history.history['val_accuracy']):.4f}")
    return model, history

def train_stage_2(model, train_data, val_data):
    """階段 2: 解凍並微調整個網路"""
    
    # 解凍 ResNet50 基礎模型
    model.layers[1].trainable = True
    
    # 凍結前面的層，只微調後面的層
    for layer in model.layers[1].layers[:-cfg.MODEL.fine_tune_layers]:
        layer.trainable = False
    
    # 重新編譯 (使用更小的學習率)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.TRAINING.learning_rate_stage2),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_3_accuracy']
    )
    
    # 訓練配置
    callbacks_stage2 = [
        tf.keras.callbacks.ModelCheckpoint(
            'data/models/checkpoints/stage2_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=cfg.TRAINING.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='logs/stage2',
            histogram_freq=1
        )
    ]
    
    # 執行微調
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=cfg.TRAINING.epochs_stage2,
        callbacks=callbacks_stage2,
        verbose=1
    )
    
    print(f"階段 2 完成 - 最佳驗證準確率: {max(history.history['val_accuracy']):.4f}")
    return model, history

def load_resnet50_training_data():
    """載入 ResNet50 專用訓練資料"""
    
    # 資料路徑
    images_dir = Path("data/training/images")
    labels_file = Path("data/training/labels.csv")
    
    # 載入標籤
    labels_df = pd.read_csv(labels_file)
    
    # 載入影像
    images = []
    labels = []
    
    for _, row in labels_df.iterrows():
        img_path = images_dir / row['filename']
        if img_path.exists():
            # 載入並預處理影像
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            
            images.append(img)
            labels.append(row['label'])
    
    # 轉換為 numpy 陣列
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)
    
    # 標籤轉為 one-hot 編碼
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=10)
    
    # 分割資料集
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels_onehot, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42
    )
    
    # 建立資料擴增生成器
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        **cfg.DATA_AUGMENTATION
    )
    
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input
    )
    
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input
    )
    
    # 建立資料生成器
    train_generator = train_datagen.flow(
        train_images, train_labels,
        batch_size=cfg.MODEL.batch_size,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        val_images, val_labels,
        batch_size=cfg.MODEL.batch_size,
        shuffle=False
    )
    
    test_generator = test_datagen.flow(
        test_images, test_labels,
        batch_size=cfg.MODEL.batch_size,
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def evaluate_resnet50_model(model, test_data):
    """全面評估 ResNet50 模型"""
    
    print("📊 開始模型評估...")
    
    # 基本評估
    test_loss, test_accuracy, test_top3_accuracy = model.evaluate(
        test_data, verbose=1
    )
    
    print(f"測試集準確率: {test_accuracy:.4f}")
    print(f"Top-3 準確率: {test_top3_accuracy:.4f}")
    
    # 詳細預測分析
    predictions = model.predict(test_data, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    
    # 獲取真實標籤
    y_true = []
    for i, (_, labels_batch) in enumerate(test_data):
        y_true.extend(np.argmax(labels_batch, axis=1))
        if i >= len(test_data) - 1:  # 確保覆蓋所有資料
            break
    
    y_true = np.array(y_true[:len(y_pred)])  # 確保長度一致
    
    # 分類報告
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n分類報告:")
    print(classification_report(y_true, y_pred, digits=4))
    
    # 混淆矩陣
    cm = confusion_matrix(y_true, y_pred)
    print("\n混淆矩陣:")
    print(cm)
    
    # 每個類別的準確率
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    print("\n各數字識別準確率:")
    for digit in range(10):
        print(f"數字 {digit}: {class_accuracies[digit]:.4f}")
    
    # 信心度分析
    confidence_scores = np.max(predictions, axis=1)
    print(f"\n平均信心度: {np.mean(confidence_scores):.4f}")
    print(f"信心度標準差: {np.std(confidence_scores):.4f}")
    
    # 低信心度樣本分析
    low_confidence_indices = np.where(confidence_scores < cfg.MODEL.confidence_threshold)[0]
    print(f"低信心度樣本數量: {len(low_confidence_indices)} ({len(low_confidence_indices)/len(y_pred)*100:.2f}%)")
    
    evaluation_results = {
        'test_accuracy': test_accuracy,
        'test_top3_accuracy': test_top3_accuracy,
        'test_loss': test_loss,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_true, y_pred, output_dict=True),
        'class_accuracies': class_accuracies,
        'mean_confidence': np.mean(confidence_scores),
        'low_confidence_ratio': len(low_confidence_indices)/len(y_pred)
    }
    
    return evaluation_results
```

### ResNet50 超參數調整
```python
def resnet50_hyperparameter_tuning():
    """ResNet50 超參數調整"""
    
    import optuna
    
    def objective(trial):
        """Optuna 目標函式"""
        
        # 超參數搜尋空間
        learning_rate_stage1 = trial.suggest_float('lr_stage1', 1e-4, 1e-2, log=True)
        learning_rate_stage2 = trial.suggest_float('lr_stage2', 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        fine_tune_layers = trial.suggest_int('fine_tune_layers', 10, 50)
        dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.7)
        
        # 建立模型
        model = create_resnet50_with_params(
            dropout_rate=dropout_rate,
            fine_tune_layers=fine_tune_layers
        )
        
        # 載入資料
        train_data, val_data, _ = load_resnet50_training_data(batch_size=batch_size)
        
        # 訓練模型
        try:
            # 階段 1
            model.layers[1].trainable = False
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_stage1),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            model.fit(train_data, validation_data=val_data, epochs=5, verbose=0)
            
            # 階段 2
            model.layers[1].trainable = True
            for layer in model.layers[1].layers[:-fine_tune_layers]:
                layer.trainable = False
                
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_stage2),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            history = model.fit(train_data, validation_data=val_data, epochs=10, verbose=0)
            
            # 回傳最佳驗證準確率
            best_val_accuracy = max(history.history['val_accuracy'])
            return best_val_accuracy
            
        except Exception as e:
            print(f"訓練失敗: {e}")
            return 0.0
    
    # 建立研究
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print("最佳超參數:")
    print(study.best_params)
    print(f"最佳驗證準確率: {study.best_value:.4f}")
    
    return study.best_params
```
