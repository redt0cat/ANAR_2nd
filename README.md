<div align="center">

<img src="./assets/cover.png">

# Auto-NovelAI-Refactor

✨ 基於 NovelAI 的自動化繪圖工具 ✨

> 本專案可能存在一些 bug，但我會盡我所能修復它們。

</div>

## 📖 介紹

如名稱所示，這是一個基於 NovelAI 的自動繪圖工具。

除了基本的批量文生圖外，還支援了多種實用功能。

## 🗓️ 近期變更

### 新增

- 🎨 支援 NAI4.5 角色參考
- 🎨 支援法術解析圖片反推 Tagger

### 最佳化

- 🔧 使用 pydantic 做配置管理

## ⚙️ 功能

- [x] ✅ 文生圖
- [x] ✅ 圖生圖
- [x] ✅ 角色分區
- [x] ✅ 風格遷移
- [x] ✅ 導演工具
- [x] ✅ 超分降噪
- [x] ✅ 法術解析
- [x] ✅ 圖片篩選
- [x] ✅ 插件系統

## 🔧 部署

### 1. 安裝 Python

到 [Python 官網](https://www.python.org/downloads/) 下載 Python（推薦 3.10+）。安裝時勾選 `Add to PATH` 選項。

### 2. 安裝 Git

到 [Git 官網](https://git-scm.com/downloads) 下載 Git。

### 3. 複製專案

```
git clone https://github.com/zhulinyv/Auto-NovelAI-Refactor
```

如有困難請參考 [自述檔案](https://github.com/zhulinyv/Semi-Auto-NovelAI-to-Pixiv#%EF%B8%8F-%E9%85%8D%E7%BD%AE)

### 4. 設定 Token

將 `token` 填入 `.env` 檔案

### 5. 啟動

Windows 使用者可以雙擊 `run.bat` 啟動。

其他系統：

```
pip install -r requirements.txt
python main.py
```

## 💡 貢獻

歡迎提交 PR 和 Issue！

## 📜 授權條款

本專案基於 [GPL-3.0](./LICENSE) 授權條款。

## 🙏 致謝

感謝 [NovelAI](https://novelai.net/) 提供的優質服務！

感謝所有貢獻者！

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=zhulinyv/Auto-NovelAI-Refactor&type=Date)](https://star-history.com/#zhulinyv/Auto-NovelAI-Refactor&Date)
