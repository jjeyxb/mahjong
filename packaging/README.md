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
  --browser-root /path/to/ms-playwright \
  --public-key /path/to/update-public-key.txt
```

Windows 需使用 Windows x64 的 Python、CPU 版 torch 和已編譯的 `libriichi.pyd`。
Mac 需使用 arm64 Python 與 `libriichi.so`。不能跨 OS 建置。
產物位於 `dist/<平台>/MIA/`。沒有指定公鑰的內測包會明確停用網路更新驗證，
不會接受未簽署套件。CUDA／ROCm 元件沒有內建在基礎包；尚無已發布的 GPU 元件。

Windows 可執行 `packaging/windows/build.ps1 -AssetSource ... -BrowserRoot ...`，
再以 Inno Setup 編譯 `packaging/windows/mia.iss`；這些腳本尚未在 Windows 驗收。
Mac 可用 `python tools/package_dmg.py dist/macos-arm64/MIA dist/MIA.dmg`
建立僅含 App 與安裝識別的內測 DMG，不會把原型使用時的錄影或登入資料打進去。
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
封裝只能包含 `program/` 或 `MIA.app/`，保留旁邊的 `userdata/`。
新版本須回報 UI 啟動成功，失敗將舊程式移回；下載與安裝紀錄在 `updates/`、`userdata/logs/`。

先在另行保管的位置產生金鑰；不要把私鑰放進發布目錄、安裝包或 Git：

```sh
python tools/release.py keygen --private /private/location/release.pem --public update-public-key.txt
python tools/release.py create --source dist/macos-arm64/MIA/MIA.app \
  --out out/release --private /private/location/release.pem \
  --version 0.1.1 --platform macos-arm64 \
  --url https://github.com/jjeyxb/mahjong/releases/download/v0.1.1/payload.zip
```

發布工具只建立本機檔案，不會上傳。基礎包內的版本與公鑰需與準備發布的版本一致。
更新 URL 尚未對外建立，沒有設定正式私鑰；不要將測試金鑰當作正式發布身份。

元件管理器接受獨立的 `component/` 封裝，要求簽署資訊與 App 版本完全相符，
worker 的 `--self-check` 必須實際通過且回報正確後端，才會寫入 `components/active.json`。
CUDA／ROCm 的真正硬體偵測、套件建置及機器驗收仍需對應硬體；沒有相容元件時沿用基礎 worker。
目前不提供超過 GitHub 單檔限制的更新 archive，自動分片下載待後續實作。

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
安裝版擷取子程序的 stdout／stderr 會保存在 `userdata/logs/capture.log`，
失敗時狀態列提供該路徑。離線包的 headless 擷取也使用包內完整 Chromium。

Mac 的圖像辨識需要授權 **MIA.app 本身** 的螢幕錄製權限；開發時給終端機或
VS Code 的權限不會自動給安裝版。開啟畫面辨識或依賴畫面的 AI 建議／放銃分析時，
UI 會在啟動工作執行緒及載入模型前請求系統授權。仍未授權就顯示含「開啟系統設定」
與「稍後」按鈕的提醒框，功能保持關閉；不會在背景重複彈窗。純封包模式不受影響。
後端也會在搜尋視窗前檢查權限，避免只顯示「找不到遊戲視窗」。
若系統設定清單沒有 MIA，可在「隱私權與安全性 → 螢幕錄製」按 + 加入安裝
資料夾裡的 MIA.app；授權後完整退出再重開。對同一後端只主動請求一次。

更新 helper 已以本機簽署的合成更新測試成功替換與失敗復原，
但沒有發布 GitHub 更新檔，也未用此測試宣稱已驗證遠端正式更新通道。
