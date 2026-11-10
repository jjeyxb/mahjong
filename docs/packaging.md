# MIA 安裝、更新與解除安裝研究

研究與實作日期：2026-11-10。已完成第一版打包與更新原型；正式發布仍需 Windows 原機驗收、發布金鑰及 Mac 簽章／公證。

## 本次實作進度

- 新增 `mia.distribution`：開發／封裝路徑、更新驗證、下載、解包、安裝、啟動檢查、失敗復原與可選元件管理。
- 主程式新增「說明 → 檢查更新」與 GPU 元件入口，背景下載不阻擋辨識；安裝需要確認並先停止使用中的功能。
- 新增 `tools/package.py`、`tools/package_dmg.py`、`tools/release.py`，以及 Windows 的建置與 Inno Setup 腳本。發布工具只產生本機檔案，不會上傳。
- Mac 原型保留獨立的主程式／Mortal／YOLO／更新程序，內含模型及 Chromium。Mortal、MPS 模型推論與介面已進行本機檢查。
- 實作先採共用的獨立更新 helper，而非 Sparkle；封裝只替換 `MIA.app`／`program`，成功啟動才更新版本紀錄。正常失敗可復原，突然斷電的自動恢復仍需進一步驗收。
- 無正式發布公鑰時，更新入口會明確告知尚未設定，不會接受未驗證的更新。GPU 加速包的安裝與回退介面已備妥，尚未建置或發布 CUDA／ROCm 元件，也未宣稱 AMD 已支援。
- 保留既有資料的相對布局，例如登入資料實際位於 `userdata/live/chrome-profile`。下方目錄樹為規劃示意；自有資料都留在同一安裝根目錄。

建置、簽署更新及目前限制見 [packaging/README.md](../packaging/README.md)。原型以獨立 worktree 開發，已同步 Claude 最後提交的 `7294e9f`；沒有回復或刪除它的辨識修改。

## 已確認的需求

- 首版支援 Windows x64 與 Apple Silicon Mac（arm64）。
- 維持 PyTorch／YOLO，不轉 ONNX。
- 基礎安裝包包含必要環境與模型，首次安裝與基礎推論不要求連網、Python、pip、Homebrew 或 Rust。Windows 提供 CPU 基礎環境；Apple Silicon Mac 保留已驗證的 MPS 路徑並可選 CPU。
- CUDA／ROCm 等可選 GPU 加速元件不必放進基礎安裝包，使用時由 App 下載至同一安裝根目錄。首次取得加速包需要網路，已安裝且相容的加速包可離線使用。這項需求取代先前「所有加速環境都隨離線包提供」的假設。
- 使用者可指定安裝位置；應用程式自有的環境、模型、設定、錄影與紀錄集中於一個專用根目錄。
- App 內下載更新，使用者確認後關閉程式並安裝。
- 更新檔透過 GitHub Releases 公開下載。
- 發布者目前尚未確認持有 Apple Developer 付費帳號或 Windows 程式碼簽章憑證。

## 結論與建議順序

首版必須包含更新的入口與信任基礎，但先確定可搬移的安裝結構、資源路徑和子程序啟動方式，再實作平台更新流程。不能只在現有開發版加上「下載新版」，就視為可發布。

建議以 **PyInstaller 的資料夾模式**進行原型：主程式、Mortal 與 YOLO 分別在自己的環境建置，保留目前的程序與 IPC 邊界。封包擷取可使用主程式的專用子命令，或另外建置擷取 helper。使用者不需要另外安裝 Python；Windows 與 macOS 必須各自建置與測試。[PyInstaller 官方說明](https://pyinstaller.org/en/stable/operating-mode.html)

這是根據目前程式結構提出的選擇，尚未驗證 PySide6、PyObjC、兩套 torch、Ultralytics、Playwright 與 libriichi 的完整封裝。需先做可啟動及可推論的原型，再鎖定建置工具版本與規格。

## 目前專案需要先調整的地方

| 接點 | 目前狀態 | 發布版所需調整 |
| --- | --- | --- |
| `src/mia/utils/paths.py` | 從 `__file__.parents[3]` 推算 repo 根目錄 | 分開定義唯讀程式資源與可寫使用者資料；不能將 frozen 的內部資源位置當成安裝根目錄 |
| `src/mia/engine/mortal.py` | 尋找 `.venv`、`bot.py` 和上游目錄裡的 libriichi | 支援包內引擎執行檔，保留開發版啟動方式 |
| `src/mia/yolo/client.py` | 尋找 YOLO `.venv`，執行 `worker.py` | 支援包內 YOLO 執行檔與明確的模型／素材路徑 |
| `src/mia/live/source.py` | `sys.executable tools/gt.py ...` | frozen 後 `sys.executable` 是 App，需改成真正支援的擷取子命令或 helper |
| `config/default.local.yaml`、`data/ui_state.json` | 與專案資料混放 | 預設值隨程式更新，個人覆寫與 UI 狀態放在 `userdata/` |
| `src/mia/ui/theme.py` | 產生圖示至系統暫存目錄 | App 自有快取改用安裝根目錄下的快取位置 |
| Playwright | 瀏覽器通常位於使用者的共用快取 | 建置時收集指定版本 Chromium，執行時指定包內瀏覽器路徑與自己的 profile |
| 版本資訊 | `pyproject.toml` 目前為 `0.1.0` | 發布版本、程式識別、架構及模型／執行環境相容資訊集中管理 |
| 可選 GPU 加速 | 尚未實作元件下載、安裝或切換；AMD 未完成整合 | 以版本化的完整推論元件提供對應 PyTorch 與加速函式庫，驗證成功後才啟用 |

PyInstaller 提供 frozen 狀態、包內資源位置與執行檔位置等資訊，路徑層需明確區分這些概念。[執行期路徑文件](https://pyinstaller.org/en/stable/runtime-information.html)

三個開發 `.venv` 的 Python 都連到這台 Mac 的 `/opt/homebrew/Cellar/...`，不能直接複製交付。Python 官方也明確將 venv 視為不可直接搬移的環境。[venv 文件](https://docs.python.org/3/library/venv.html)

目前主程式使用 Python 3.14；Mortal 與 YOLO 使用 3.12，但 torch、numpy 的需求不同，首版保持獨立封裝。使用者不需要安裝 Rust；發布端預先編譯各平台的 libriichi，並收集原生相依套件。

## 安裝目錄

Windows 的概念配置如下，名稱可在原型完成後調整：

```text
使用者指定的位置/MIA/
  program/                 主程式、內含 Python 的引擎、模型、瀏覽器、素材
  components/              按需下載的加速推論環境，依平台／後端／版本隔離
  userdata/
    config/                個人設定、校正、UI 狀態
    browser-profile/       登入狀態與瀏覽器自有資料
    recordings/            使用者錄影、封包與辨識紀錄
    logs/
    cache/
  updates/                 下載、驗證、暫存與更新 helper
  install-info.json        安裝識別與版本，不存發布私鑰
  unins*.exe               Windows 解除安裝入口
```

這裡的 `program/` 由安裝器管理，`userdata/` 由程式管理，`components/` 由元件管理器依已驗證的版本資訊管理；更新時不拿新版本的預設檔覆寫使用者資料。正式布局需確定啟動檔位置與捷徑目標。

Windows 建議以 Inno Setup 提供選擇目的資料夾的精靈，採個人安裝模式。預設放在當前使用者可寫入的位置，也允許其他磁碟的專用資料夾。若環境和可寫資料都必須在同一處，首版不支援一般帳號無權寫入的 `Program Files` 位置，也不要求使用者平時以系統管理員啟動 App。[目的資料夾](https://jrsoftware.org/ishelp/topic_setup_defaultdirname.htm)、[個人安裝模式](https://jrsoftware.org/ishelp/topic_setup_privilegesrequired.htm)

Mac 使用相同的外層原則：

```text
使用者指定的位置/MIA/
  MIA.app/                 主程式、引擎、模型、瀏覽器、素材
  components/              預留的可選元件；首版 Mac GPU 走包內 MPS
  userdata/                設定、錄影、登入狀態、紀錄與自有快取
  updates/                 App 管理的更新暫存
  install-info.json
```

可用 DMG 提供完整的 `MIA/` 資料夾，讓使用者拖至自己選擇且可寫入的位置，例如 `~/Applications/MIA/`。首次開啟建立 `userdata/`。若要與 Windows 一樣的選路徑精靈，可另外製作小型安裝 App，但不是原型必要條件。需測試非預設位置、空白與中文路徑，以及檔案系統是否保留符號連結。

Mac 的可變資料放在 `MIA.app` 外、同一根目錄內；更新器替換 App 時保留旁邊的資料。所有會執行的程式與固定模型應納入發布版本的驗證範圍。

「同一資料夾」適用於 App 自有檔案。系統的螢幕錄製權限、顯卡驅動、應用程式註冊、安裝器及更新框架可能使用的系統暫存仍由作業系統管理，不能承諾系統其他位置完全不會產生紀錄。若選用封包 MITM 功能，其憑證與系統設定也需獨立評估清除方式；不應為安裝一般圖像辨識功能而啟用它。

## 基礎離線內容與體積

需要收集：

- 主程式、平台擷取套件、Qt plugins、OpenCV、麻將規則及封包解析所需模組。
- Mortal 推論程式、各平台 libriichi、CPU 推論所需 torch 及要提供的基準／風格權重。
- YOLO 推論程式、四個權重、旁附 JSON、模型讀圖素材及基礎 torch／torchvision；Windows 使用 CPU 版本，Mac 提供 MPS／CPU 所需套件。可選 CUDA／ROCm 環境另外下載。
- 現有樣板、HUD、圖示、協定資料及預設設定。
- 網頁封包模式所用的指定版本 Chromium 與 Playwright driver；不使用使用者登入中的瀏覽器 profile 作為發布素材。
- 專案與相依元件隨附的授權／來源說明。

Playwright 可以透過 `PLAYWRIGHT_BROWSERS_PATH` 指定安裝及執行時的瀏覽器位置；版本必須與所包的 Playwright 相符。[瀏覽器文件](https://playwright.dev/python/docs/browsers)

不打包 `downloads/`、測試錄影、封包真值、訓練資料、個人設定、登入 cookie、開發快取、Cargo 建置目錄或整份上游 Git checkout。上游中推論需要的模組與二進位檔則必須收集。

本機目前三套開發環境約為 1.9 G、689 M、869 M，模型目錄約 675 M；這些是開發環境的磁碟占用，**不是安裝包大小估計**。正式基礎包與可選加速包的壓縮體積需分別在原型完成後量測；GPU 元件下載時也應顯示所需空間。

GitHub Releases 每個 asset 必須小於 2 GiB。若 Windows 基礎安裝包超限，可用 Inno Setup 的 `Setup.exe` 加多個 `Setup-*.bin`，每片留在限制以內；全部檔案下載到同一資料夾後可離線安裝，App 更新流程也需下載並驗證全部分片。獨立加速包若超限，同樣需以可驗證的分片提供，不能將過大的單一檔案直接上傳。[GitHub 限制](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas)、[Inno 分包](https://jrsoftware.org/ishelp/topic_setup_diskspanning.htm)

Mac 若使用 Sparkle 的標準單一 archive 流程，也必須先確認更新檔能低於 GitHub 的限制。若超過，需要另外設計分片下載／組裝與安裝交接，或調整發布內容布局；不能假設 Sparkle 自動支援任意分片。本項應在選定 Mac 更新器前用實際產物判定。

## 可選 GPU 加速包

基礎版安裝後即可使用。App 偵測硬體與驅動，僅對支援且已測試的平台提供可選加速包，顯示下載大小與空間需求，由使用者選擇下載。取消、斷網或不相容時保留基礎環境可用；CPU 的即時效能仍需測試，不能因為能執行就宣稱與 GPU 一樣快。

下載內容應是一套**與 App 相容的 GPU 推論環境**，包括推論 worker、對應 PyTorch／torchvision 與必要的加速函式庫，不是只把 CUDA DLL 放到 CPU 版 PyTorch 旁邊。這也適合目前將 AI 保持在獨立子程序的架構：基礎 worker 與可選 GPU worker 各自封裝，由啟動層選擇，避免改寫 frozen 主程式內的套件。[PyTorch 平台與計算後端選擇](https://pytorch.org/get-started/locally/)

例如可安裝於 `components/yolo/windows-x64/cuda/<元件版本>/`；實際元件需記錄支援的 App／IPC 版本、模型組、OS、架構、GPU／驅動條件、Python 與 torch 版本。Mortal 目前仍走 CPU；YOLO 的 GPU 元件不自動套用到 Mortal。

先下載至 `updates/`，驗證發布簽章與各檔案完整性，完成解包及實際模型推論檢查，再於停止相關 worker 後切換啟用。若驗證或啟用失敗，繼續使用原來的 worker。更新 App 時重新核對已下載元件的相容性；不相容的舊元件不強行載入，也不在使用者電腦執行 pip 將套件升到最新版。

ROCm 元件需另外完成整合與硬體驗收；不能只偵測到 AMD 品牌就下載。其支援範圍取決於 OS、GPU、驅動與框架組合，應按發布時的官方支援表與專案測試結果建立可用清單。不支援的機器仍提供 CPU。首版 Apple Silicon Mac 的 GPU 使用包內 MPS，不需要 CUDA／ROCm。[AMD Windows 支援表](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)

本方案只將 App 自有的使用者空間推論套件放入安裝目錄，不隨意安裝整套系統 SDK、修改全域 PATH 或替換系統顯卡驅動。必要的系統驅動仍由對應廠商與作業系統管理。解除安裝時移除 MIA 的 `components/`，不移除使用者的系統驅動。

## 第一版必須有的更新基礎

1. 固定的應用程式識別（Windows AppId、Mac bundle identifier）、版本與平台架構。
2. 版本資訊頁與「檢查更新」，包含可取消下載、下載進度與錯誤處理。
3. 更新資訊的公開來源及預先放入 App 的驗證公鑰。
4. 套件版本、下載檔名、大小、雜湊與程式／模型／執行環境相容資訊。
5. 更新後保留的資料邊界、設定遷移與失敗復原策略。

建議流程：

```text
檢查相符平台的新版本
  → 顯示更新內容與大小
  → 下載至 updates/（中斷時舊版仍可使用）
  → 驗證發布簽章、大小與完整性
  → 使用者確認安裝；有正在進行的對局／錄製則先等待
  → 儲存設定，正常停止 MIA 自己的擷取與 AI 子程序
  → 由獨立 helper／平台更新框架執行安裝
  → 啟動新版本並檢查；失敗保留可復原的舊版本與設定
```

雜湊用於檢查完整性，發布者簽章用於驗證來源。私鑰不放在 App 或公開倉庫；公開下載不應要求把個人 GitHub token 包進程式。若未來更換驗證公鑰，舊版也需要可驗證的金鑰轉換流程。

Windows 建議在 App 內下載並驗證完整安裝器，由獨立 helper 等主程式與相關子程序退出，再執行同一套 Inno 升級安裝。helper 不放在當次要覆寫的 `program/` 裡。固定 AppId 讓後續版本延續同一份安裝識別與解除安裝紀錄；這不等於已經具備自動回滾，復原仍需另外設計與測試。[AppId 文件](https://jrsoftware.org/ishelp/topic_setup_appid.htm)

Mac 優先評估 Sparkle 2 管理下載、確認與替換 App。MIA 是 PySide6，需用原生橋接或小型原生 helper 整合，不是直接加一個 pip 套件即可。完整 App、內含 helpers／動態函式庫需完成簽署，再產生可驗證的更新 archive；是否採用此方案仍受前述體積測試影響。[Sparkle 設定](https://sparkle-project.org/documentation/)、[程式化整合](https://sparkle-project.org/documentation/programmatic-setup/)

首版主程式更新將相容的程式、基礎引擎和模型作為一組發布；可選加速包另以明確的相容版本管理，已安裝且仍相容的元件可保留。先不提供獨立替換某個 `.pt` 的功能。之後若要進一步降低下載量，再增加按版本驗證的模型元件更新。

## 簽章與首次發布

可先製作內測產物，不必立刻購買或申請帳號。正式發給一般使用者前：

- Mac 建議申請 Apple Developer Program，使用 Developer ID 簽章並完成公證與離線驗證所需的 stapling；沒有這些準備時，不能承諾下載後直接開啟或離線首次啟動都不受 Gatekeeper 阻擋。[Developer ID](https://developer.apple.com/help/account/certificates/create-developer-id-certificates)
- Windows 可先做未簽名內測包；公開安裝器及解除安裝器建議加入可信的程式碼簽章。簽章不保證完全沒有 SmartScreen 提示。[Inno 簽章解除安裝器](https://jrsoftware.org/ishelp/topic_setup_signeduninstaller.htm)
- 自己的更新驗證金鑰與作業系統的程式碼簽章是不同用途；不能互相取代。Sparkle 使用 EdDSA 簽署發布更新。[Sparkle 更新簽署](https://sparkle-project.org/documentation/#3-segue-for-security-concerns)

目前尚未產生金鑰、購買憑證、設定發布權限、建立 Release 或上傳檔案。

## 解除安裝

Windows 提供系統「已安裝的應用程式」與資料夾內的解除安裝入口，清除程式與已下載的 MIA 加速元件；Mac 可先以移除專用 `MIA/` 根目錄完成主要檔案清除，若有需要再提供資料管理／解除安裝 helper。

建議預設保留設定與錄影，提供明確選項一併刪除；使用者選擇全刪時，僅刪除這次安裝所擁有的資料子目錄。不能直接使用萬用字元刪除整個任意目的資料夾，安裝時也應避免把磁碟根目錄或其他程式的目錄當作專用根目錄。[Inno 官方的解除安裝注意事項](https://jrsoftware.org/ishelp/topic_uninstalldeletesection.htm)

不解除安裝使用者原本的 Python、Chrome、顯卡驅動，也不刪除共用快取。App 建立的捷徑與安裝紀錄由相應安裝器清除；作業系統保留的權限與歷史紀錄不屬於自有環境。

## 實作與驗收階段

1. **路徑與啟動接點**：建立開發／發布共用的資源與資料路徑層，完成主程式、Mortal、YOLO、擷取的發布啟動入口。保留原有辨識演算法。
2. **兩平台基礎包與加速包原型**：在對應 OS／架構建置，於未安裝開發工具的乾淨電腦測試。分別量測基礎包與加速包體積，驗證離線基礎推論、相容硬體偵測、元件下載與 worker 切換，再決定 GitHub 分包與 Mac 更新器可行性。ROCm 未通過驗收前不列為已支援。
3. **更新發布原型**：本機準備舊、新兩版及測試 feed，驗證來源簽章、下載取消／中斷、空間不足、關閉程序、安裝、重啟、已下載元件相容性與失敗復原。不必先公開發布。
4. **安裝與解除安裝**：測試自選磁碟、中文／空白路徑、一般帳號、資料保留與全刪，確認只操作 MIA 擁有的檔案。
5. **正式發布**：確定支援的最低 OS／GPU、版本鎖定與 build manifest；完成簽章、Mac 公證及兩平台原機推論／擷取測試，再發布 GitHub Releases。

驗收至少包含：離線首次安裝與基礎推論；無系統 Python 時啟動；樣板／YOLO／混合模式及 Mortal 推論；封包模式的包內瀏覽器；加速包下載／中斷／重試／驗證／切換；已下載相容元件離線使用；不相容 GPU 或驅動下保留 CPU 可用；Mac MPS 與螢幕擷取權限；更新後設定與錄影保留；舊加速包相容性判定；解除安裝能移除 App 自有元件；被中斷的更新不損壞仍可使用的版本。 frozen 子程序可能繼承包內函式庫搜尋路徑，須驗證不同 Python／torch 的子程序隔離。[PyInstaller 子程序注意事項](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html#launching-external-programs-from-the-frozen-application)

PyInstaller 僅安裝於獨立建置環境；封裝與更新的修改不包含辨識演算法。首次掃描 Ultralytics 時曾觸發可選 `lap` 套件安裝，當次新增的套件已移除還原；後續建置已禁用自動安裝，使用獨立 venv、唯讀依賴來源與各角色獨立快取。正式乾淨系統驗收與發布尚未完成，不能把本機原型檢查當成所有 Windows／Mac 電腦皆已驗證。
