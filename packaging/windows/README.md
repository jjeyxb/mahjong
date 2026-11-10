# 在 Windows 上建置 MIA

Windows 版只能在 Windows x64 上建,不能從 Mac 跨平台產生。建置機需要開發工具,
**用安裝包的人不需要**:他們不必裝 Python、Rust 或任何 SDK。

## 1. 裝建置工具

- [Git](https://git-scm.com/download/win)
- [Python 3.14 與 3.12](https://www.python.org/downloads/windows/),兩個都裝,安裝時勾「py launcher」。
  主程式用 3.14,Mortal 與 YOLO 用 3.12。
- [Rust](https://rustup.rs/) 與 Visual Studio Build Tools 的「使用 C++ 的桌面開發」。只用來編 libriichi。
- [Inno Setup 6](https://jrsoftware.org/isdl.php),產生安裝檔。

裝完開一個新的 PowerShell,確認 `py -3.14 --version`、`py -3.12 --version`、`cargo --version` 都有輸出。

## 2. 取得原始碼與素材

```powershell
git clone https://github.com/jjeyxb/mahjong.git
cd mahjong

# Mortal 上游,固定在開發時用的版本
git clone https://github.com/Equim-chan/Mortal.git engines/mortal/Mortal
git -C engines/mortal/Mortal checkout 0cff2b52982be5b1163aa9a62fb01f03ce91e0d2
```

模型權重不在 git 裡。從 Mac 的專案目錄把 `models/` 整個資料夾複製到這裡的 `models/`,至少要有:

```text
models/mortal_298k.pth
models/style_call_high.pth
models/style_call_low.pth
models/style_reach_high.pth
models/style_reach_low.pth
models/yolo/region_model_best.pt
models/yolo/river_model_best.pt
models/yolo/hand_model_best.pt
models/yolo/meld_model_v32_1280.pt
models/yolo/river_model_best.infer.json
models/yolo/meld_model_v32_1280.infer.json
models/yolo/meld_model_v32_1280.crop_meta.json
```

## 3. 編 libriichi

```powershell
cd engines/mortal/Mortal
$env:PYO3_PYTHON = (py -3.12 -c "import sys; print(sys.executable)")
cargo build -p libriichi --lib --release
Copy-Item target/release/riichi.dll mortal/libriichi.pyd
cd ../../..
```

## 4. 建置

```powershell
powershell -ExecutionPolicy Bypass -File packaging/windows/build.ps1
```

腳本會在 `out/` 建三個獨立的 Python 環境、下載對應版本的 Chromium、用 PyInstaller 打包,
產物在 `dist/windows-x64/MIA/`。第一次要下載 torch 與 Chromium,約二、三十分鐘。

沒給 `-PublicKey` 的內測包會停用網路更新:「檢查更新」會說尚未設定更新來源,這是預期的。

## 5. 先不安裝,直接自我檢查

```powershell
dist/windows-x64/MIA/program/MIA.exe --self-check --engines --browser
```

輸出的 JSON 裡 `mortal_decision`、`yolo_device`、`browser_check` 都要有值。

## 6. 產生安裝檔

```powershell
& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" packaging/windows/mia.iss
```

安裝檔在 `dist/installers/`。超過 1.9 GB 會切成 `Setup.exe` 加幾個 `.bin`,放在同一個資料夾就能裝。

## 7. 驗收重點

- 安裝到自選的資料夾(含中文或空白的路徑也試一次),一般帳號,不用系統管理員。
- 網頁版:按「開始遊戲」開出內附的瀏覽器,登入後三個功能都會動。
- Steam 版:設定頁選 Steam 版,三種辨識方式都試。
- 設定頁「語言與更新」:換成 English、重新啟動,介面變英文;換回繁體中文。
- 解除安裝:從「已安裝的應用程式」移除,`userdata/` 依選擇保留或刪除。

出問題時,擷取子程序的紀錄在安裝目錄的 `userdata/logs/capture.log`。
