# 建置與內測

以獨立的 `codex/packaging` worktree 開發，已同步 Claude 完成的 `7294e9f`。
打包接點與演算法修改分開，整合時保留 Claude 的程式碼。
這是原型；正式發布金鑰、簽章、公證及 Windows 原機驗收尚未完成。

## 建置

主程式 Python 3.14，Mortal／YOLO Python 3.12，三個建置環境應彼此隔離。
可用 `.pth` 將既有依賴作為唯讀建置來源，但建置的 Python 需使用另外建立的 venv。
PyInstaller 在單獨環境安裝，`--build-tools` 指向其 `site-packages`。
建置工具禁止 Ultralytics 自動安裝，並將建置快取放在輸出目錄。

```sh
python tools/package.py \
  --asset-source /path/to/model-and-mortal-source \
  --build-tools /path/to/build-tools/site-packages \
  --mortal-python /path/to/build-mortal/python \
  --yolo-python /path/to/build-yolo/python \
  --browser-root /path/to/ms-playwright
```

更新公鑰預設用版控裡的 `packaging/release-public-key.txt`(見下面「發布金鑰」),
Windows 與 Mac 自動是同一把;`--public-key` 只在測試別的金鑰時才給。

Windows 需使用 Windows x64 的 Python、CPU 版 torch 和已編譯的 `libriichi.pyd`。
Mac 需使用 arm64 Python 與 `libriichi.so`。不能跨 OS 建置。
產物位於 `dist/<平台>/MIA/`。沒有公鑰的內測包會明確停用網路更新驗證，
不會接受未簽署套件。GPU 版 torch 不在基礎包裡，見下面「GPU 加速(Windows)」。

Windows 可執行 `packaging/windows/build.ps1 -AssetSource ... -BrowserRoot ...`，
再以 Inno Setup 編譯 `packaging/windows/mia.iss`；這些腳本尚未在 Windows 驗收。
Mac 可用 `python tools/package_dmg.py dist/macos-arm64/MIA dist/MIA.dmg`
建立內測 DMG：只有 `MIA.app`、「應用程式」捷徑與中英對照的安裝說明，不會把原型使用時的錄影或登入資料打進去。
Mac 打包版的資料放在 `~/Library/Application Support/MIA`、`~/Library/Caches/MIA`、`~/Library/Logs/MIA`，
不在 App 旁邊；Windows 照舊放在安裝資料夾的 `userdata/`。
正式發布還需使用 Developer ID 簽署完整 App、送交公證並 stapling；
目前建置腳本只做 ad-hoc 簽署。

## 發布版不含的功能

自動打牌、自己開電腦場、牌譜與自打工具只給對電腦的房間錄素材,兩種 MITM 擷取(`proxy`、`local`)
要使用者自己裝並信任 CA 憑證,這些都不隨發布版提供:

- 建置時排除 `autoplay`、`paipu`、`selfplay`、`mitmproxy` 等模組(`tools/package.py` 的 `_NOT_SHIPPED`)。
- 打包版的 `MIA --capture` 只接受瀏覽器擷取(`cdp`)與一般錄製參數,其餘一律拒絕
  (`mia.distribution.capture`)。介面的擷取模式選項也只剩 `cdp`。

## 介面語言

設定頁「語言與更新」可選繁體中文或英文,換了之後重新啟動生效。譯文在 `assets/i18n/en.json`,
以繁體中文原文為鍵;執行緒組好的狀態訊息用帶 `{名稱}` 的模板比對。介面新增文字時要同時補譯文,
`tests/unit/test_i18n.py` 會列出缺的。要加其他語言,新增一份 `assets/i18n/<代碼>.json`
並在 `mia.i18n.LANGUAGES` 加一項。

## 更新

「說明 → 檢查更新」提供背景檢查、下載、取消，以及確認後關閉安裝。
開著辨識功能或受管瀏覽器時會先要求停止，避免中斷對局。
更新 helper 驗證固定公鑰的 Ed25519 發布資訊，再核對大小與 SHA-256。
封裝只能包含 `program/` 或 `MIA.app/`，使用者資料不在封裝範圍內。
新版本須回報 UI 啟動成功，失敗將舊程式移回；下載與安裝紀錄在 `updates/` 與紀錄資料夾。
Mac 換掉的是實際執行的那個 `MIA.app`（通常在「應用程式」）；從磁碟映像執行時不更新。

### 發布金鑰

2026-11-12 產生了正式的 Ed25519 發布金鑰：

- **公鑰**:`packaging/release-public-key.txt`,進版控，打包時自動放進安裝檔。
- **私鑰**：不在專案裡。Windows 發布機放在 `%USERPROFILE%\.mia-release\mia-release-private.pem`,
  Mac 發布機放在 `~/.mia-release/mia-release-private.pem`(由開發者自己複製過去,`chmod 600`)。
  不要放進發布目錄、安裝包或 Git。**私鑰遺失 = 已安裝的玩家再也收不到自動更新**,兩台都要留、
  另外再備份一份離線的。

裝了沒有公鑰的舊內測包(v0.1.0)的人收不到自動更新，要手動下載一次新的安裝檔。

發布新版時(以 Mac 為例):

```sh
python tools/release.py create --source dist/macos-arm64/MIA/MIA.app \
  --out out/release --private ~/.mia-release/mia-release-private.pem \
  --version 0.1.1 --platform macos-arm64 \
  --url https://github.com/jjeyxb/mahjong/releases/download/v0.1.1/payload.zip
```

發布工具只建立本機檔案，不會上傳。`update-<平台>.json` 與 `payload.zip` 一起上傳到同一個
GitHub Release;App 讀的是 `releases/latest/download/update-<平台>.json`。
基礎包內的版本與公鑰需與準備發布的版本一致。單檔超過 2 GB(GitHub 限制)的更新 archive
目前不支援。

## GPU 加速(Windows)

打包版的 `MIA-yolo.exe` 只帶 CPU 版 torch。玩家在設定裡把「YOLO 運算裝置」選 GPU(或按
「說明 → GPU 加速…」)時,MIA 照安裝檔附的清單從**官方來源**下載，我們不架、也不上傳任何
GPU 元件：

| 顯示卡 | torch 來源 | 檔案數 | 下載量 |
|---|---|---|---|
| NVIDIA | download.pytorch.org(CUDA 12.8) | 40 | 2.81 GB |
| AMD RX 7000/9000 | repo.radeon.com(ROCm 7.2.1) | 43 | 1.96 GB |
| Intel Arc | download.pytorch.org(XPU) | 58 | 1.41 GB |

另外都要 python.org 的 embeddable Python 3.12.10(11 MB)與 PyPI 的 pip。

- 清單在 `packaging/gpu/`,由 `tools/lock_gpu.py` 產生：每個檔案的網址、SHA-256 與大小，
  版本照實測過的 `engines/yolo/.venv` 釘住。打包時整個資料夾放進 `resources/gpu/`。
- 玩家電腦上的 pip 以 `--require-hashes --no-deps --no-index` 執行：只裝清單上的檔案，內容不符就失敗，
  不會自己挑版本。裝在 `components/gpu/` 的暫存資料夾，實際在顯示卡上跑一次推論自我檢查，
  通過才啟用；失敗或取消只刪暫存，原本的 CPU 推論不受影響。
- 推論時用下載來的 Python 執行 `MIA-yolo/_internal/` 裡附的 `yolo.py` / `worker.py` / `agent/`,
  跟 exe 是同一份程式。解除安裝時 `components/` 一起刪。
- 換 torch 版本：改 `engines/yolo/requirements-<backend>.txt`,在開發環境實測，再跑
  `python tools/lock_gpu.py --python downloads/python-3.12.10/python.exe`,重新打包。
  清單一換，舊的 GPU 元件就不再被使用(推論退回 CPU),玩家再選一次 GPU 會照新清單重裝。
- Mac 不需要：基礎包的 torch 本來就有 MPS。

## 本機檢查

```sh
dist/macos-arm64/MIA/MIA.app/Contents/MacOS/MIA --self-check --engines --browser
```

這會讀取包內資源，以 Mortal 處理一個合成牌局、YOLO 推論一張空白畫面，
並用包內 Chromium 開啟記憶體內頁面；不會登入遊戲或啟動對局。
`--capture inspect --help` 可確認封包擷取子命令的入口。

Mac 也必須透過 Finder／`open -n -W MIA.app --args --capture ...` 驗收啟動，
因為 Finder 的工作目錄可能是唯讀的 `/`。只在開發目錄執行自我檢查不會
發現 CLI 匯入模組時意外建立相對路徑檔案的問題。可指定 `data:text/html,...`
測試頁與獨立暫存 profile，再以 `--duration 2` 自動結束，不必開啟真正對局。
安裝版擷取子程序的 stdout／stderr 會保存在紀錄資料夾的 `capture.log`（Windows `userdata/logs/`，Mac `~/Library/Logs/MIA/`），
失敗時狀態列提供該路徑。離線包的 headless 擷取也使用包內完整 Chromium。

Mac 的圖像辨識需要授權 **MIA.app 本身** 的螢幕錄製權限；開發時給終端機或
VS Code 的權限不會自動給安裝版。開啟畫面辨識或依賴畫面的 AI 建議／放銃分析時，
UI 會在啟動工作執行緒及載入模型前請求系統授權。仍未授權就顯示含「開啟系統設定」
與「稍後」按鈕的提醒框，功能保持關閉；不會在背景重複彈窗。純封包模式不受影響。
後端也會在搜尋視窗前檢查權限，避免只顯示「找不到遊戲視窗」。
若系統設定清單沒有 MIA，可在「隱私權與安全性 → 螢幕錄製」按 + 加入「應用程式」
裡的 MIA.app；授權後完整退出再重開。對同一後端只主動請求一次。

更新 helper 已以本機簽署的合成更新測試成功替換與失敗復原，
但沒有發布 GitHub 更新檔，也未用此測試宣稱已驗證遠端正式更新通道。
