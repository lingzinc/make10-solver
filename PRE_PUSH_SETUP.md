# Git Pre-push Hook 設定指南
# 適用於 Windows 11 + uv 環境

## 設定完成確認

您的 pre-push hook 已經設定完成！

- **主要 Hook**: `.git/hooks/pre-push` (Git 執行的入口點)
- **PowerShell 實作**: `.git/hooks/pre-push.ps1` (實際的測試邏輯)

Hook 架構：Git → shell 橋接器 → PowerShell 腳本

## 工作原理

當您執行 `git push` 時，Git 會自動執行 pre-push hook：

1. ✅ 檢查 uv 是否可用
2. ✅ 檢查 pytest 是否安裝
3. ✅ 執行所有測試 (`uv run pytest tests/ -v --tb=short`)
4. ✅ 如果測試通過，允許 push
5. ❌ 如果測試失敗，阻止 push

## 使用方式

### 正常情況
```bash
git add .
git commit -m "feat: 新功能"
git push origin master
```

如果測試通過，push 會正常執行。

### 測試失敗時
```bash
git push origin master
# 輸出：
# 🧪 執行 pre-push hook - 開始執行單元測試...
# 📋 執行單元測試...
# ❌ 單元測試失敗！
# 🚫 阻止 push - 請修復測試錯誤後再次嘗試
```

### 緊急繞過（不建議）
```bash
git push --no-verify origin master
```

## 測試 Hook

手動測試 hook 是否工作：

```bash
# 測試完整的 hook 流程（推薦）
git push --dry-run origin master

# 或直接測試 PowerShell 腳本
powershell -ExecutionPolicy Bypass -File .git/hooks/pre-push.ps1
```

## 疑難排解

### 問題 1: Hook 沒有執行
- 確認檔案位置：`.git/hooks/pre-push` (入口點) 和 `.git/hooks/pre-push.ps1` (實作)
- 確認 PowerShell 執行原則：`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 問題 2: uv 命令找不到
- 確認 uv 已安裝：`uv --version`
- 確認 PATH 設定正確

### 問題 3: pytest 找不到
```bash
uv add --dev pytest
```

## 自訂設定

您可以修改 `.git/hooks/pre-push` 來：

1. **更改測試範圍**：
   ```bash
   uv run pytest tests/test_specific.py  # 只測試特定檔案
   ```

2. **添加程式碼格式檢查**：
   ```bash
   uv run black --check .
   uv run flake8 .
   ```

3. **添加類型檢查**：
   ```bash
   uv run mypy src/
   ```

## 團隊共享

要讓團隊成員也使用相同的 hook：

1. 將 hook 檔案移到專案根目錄
2. 建立安裝腳本
3. 在 README 中說明如何設定

範例安裝腳本：
```bash
#!/bin/bash
cp hooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push
echo "Pre-push hook 安裝完成！"
```
