# Majsoul Copilot

跨平台（macOS / Windows）的雀魂麻將輔助系統。透過螢幕擷取與電腦視覺解析牌桌狀態，
轉換為 MJAI 事件流餵入本機端的 Mortal 麻將 AI，並以 HUD 呈現打牌建議。

大學畢業專題。

---

## 專題定位

本專案的技術主軸是**「電腦視覺能不能可靠地重建一個即時對局的完整狀態」**，
而不只是「做一個能跑的外掛」。

因此架構上刻意做了一件事：另外實作一條 WebSocket 封包解析路徑作為
**Ground Truth（GT）**。GT **不參與線上決策**，只用於兩件事：

1. 自動產生帶標註的訓練／測試資料集（畫面 ↔ 完美狀態的成對資料）
2. 量化 CV 的辨識準確率（per-tile accuracy、事件流 F1、混淆矩陣）

這讓「準確率」從一句宣稱變成一組**可重現的實驗數據**，也是本專題最主要的貢獻。

### 關於「100% 準確率」

純視覺辨識無法在數學上保證 100%。但架構可以做到**讓錯誤不累積**，設計上有三道防線：

| 層 | 職責 | 原則 |
|---|---|---|
| `vision/` | 回答「這一幀我看到什麼」 | **完全無狀態**，絕不推論發生了什麼事件 |
| `tracker/` | 跨幀 diff 產生事件 | 以麻將規則校驗（張數守恆、同牌 ≤4、副露來源合法） |
| `tracker/recovery.py` | 偵測到不一致時重新同步 | 寧可重建狀態，也不帶著錯誤繼續跑 |

`Observation`（單幀快照）與 `GameState`（跨幀狀態機）分離是整個設計的核心。
把這兩者混在一起，是這類專案最常見的失敗原因。

---

## 架構

### 三進程模型

```
┌───────────────────────────────────────────────┐
│  主程式   (Python 3.14 · .venv)                │
│  capture → vision → tracker → mjai → ui        │
└────────┬───────────────────────────┬──────────┘
         │ MJAI JSON-lines / stdio   │ JSON-lines
         ▼                           ▼
┌──────────────────────┐   ┌──────────────────────────┐
│  AI Engine × N       │   │  GT 擷取(二選一)          │
│  (Python 3.12)       │   │  CDP / mitmproxy          │
│  Mortal + libriichi  │   │  只寫 .jsonl 錄影檔        │
└──────────────────────┘   └──────────────────────────┘
```

**為什麼 AI 引擎必須是獨立子程序：**

1. `libriichi` 不在 PyPI（需從 Mortal repo 自行編譯），`mjai` 在 PyPI 上只有 `cp312` wheel
   —— 兩者都裝不進主程式的 Python 3.14 環境。
2. torch 載入慢、崩潰時不該拖垮 UI。
3. 三方比較 = 同時啟動三個引擎子程序，餵同一條事件流。
4. **AGPL 隔離**：Mortal 是 AGPL-3.0，透過子程序 + 明確定義的 IPC 協定通訊，
   通常視為獨立程式，比直接 `import libriichi` 大幅降低授權傳染風險。

MJAI 本來就是 JSON-lines over stdio 的協定，子程序化是它的原生形式，不是妥協。

### 資料流

```
Frame ──vision──▶ Observation ──tracker──▶ GameState ──encoder──▶ MjaiEvent
(ndarray)         (單幀,無狀態)             (跨幀狀態機)            (JSON)
                                                                      │
                            Advice ◀────── engine × N ◀───────────────┘
                              │
                              ▼
                          ViewModel ──▶ Panel（側邊視窗）
                                     └─▶ Overlay（透明置頂）
```

兩種 HUD 共用同一個 `ViewModel`，不重複實作顯示邏輯。

### 目錄

```
src/majsoul_copilot/
├── config/        設定模型 (pydantic)、YAML 載入、skin profile
├── capture/       平台擷取抽象:macos(Quartz) / windows(PrintWindow) / mss fallback
├── calibration/   牌桌矩形偵測、DPI 與 Retina 縮放、ROI 正規化
├── vision/        純 CV,無狀態:tiles/ hand meld river dora indicators
├── tracker/       state / differ / validator / recovery
├── mjai/          events / encoder / tiles(牌表示轉換)
├── engine/        base / subprocess_engine / mortal / dummy / multiplex
├── ui/            viewmodel + panel/ overlay/ widgets/
├── groundtruth/   cdp / capture_addon / dump / liqi / schema / to_mjai
├── eval/          align / metrics / report
├── recorder/      錄製資料集、離線回放
└── utils/

engines/mortal/    Mortal 子程序的獨立環境 (Python 3.12)
tools/             開發用 CLI:模板擷取器、ROI 標註器、回放器
assets/tiles/      牌面模板圖,依 skin profile 分目錄
config/            ROI 定義、skin profile YAML
data/              錄製資料集 (gitignore)
models/            權重 .pth (gitignore)
tests/fixtures/    靜態截圖 + 期望輸出
```

---

## 安裝

### 1. 主程式環境（Python 3.14）

```bash
git clone <repo> && cd mahjong
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt          # 或 requirements-dev.txt
```

### 2. AI 引擎環境（Python 3.12，獨立）

見 [`engines/mortal/requirements.txt`](engines/mortal/requirements.txt) 內的建置說明。
需要自行從 [Mortal repo](https://github.com/Equim-chan/Mortal) 編譯 `libriichi`：
`cargo build -p libriichi --lib --release`，再把產物複製成 `mortal/libriichi.pyd`（Windows）
或 `libriichi.so`（Linux）。權重 `.pth` 放到 `models/`（已 gitignore）。

### 3. 平台權限

- **macOS**：首次執行需在「系統設定 → 隱私權與安全性 → 螢幕與系統錄製」勾選
  執行本程式的應用程式（從終端機跑就是終端機 App，從 VS Code 跑就是 VS Code）。
  **授權後必須完全結束並重新啟動該應用程式**，權限才會生效。
  未授權時的徵兆是：視窗列得出來，但標題全是空的，且擷取回傳 None。
- **Windows**：程式啟動時會設定 `PER_MONITOR_AWARE_V2` DPI awareness，
  否則座標會被系統縮放偷改。

### 4. 驗證安裝

```bash
python tools/capture_probe.py --list                     # 看得到哪些視窗
python tools/capture_probe.py --out shot.png             # 抓一張雀魂畫面
python tools/capture_probe.py --calibrate --out cal.png  # 校正並輸出標註圖
python tools/capture_probe.py --bench 60                 # 測擷取幀率
```

`--calibrate` 產生的標註圖中，綠框是偵測到的牌桌矩形、橘線是三分格線。
請目視確認綠框有貼齊遊戲畫布——後續所有 ROI 都以這個矩形為基準，它錯了全部都會錯。
若自動偵測不準，在 `config/default.local.yaml` 以 `manual_table_rect` 手動指定。

### 5. 錄製資料集

```bash
python tools/record.py --duration 60          # 錄 60 秒
python tools/record.py                        # 錄到 Ctrl-C
python tools/record.py --inspect data/recordings/<id>    # 檢視
python tools/record.py --export data/recordings/<id> --export-dir out/  # 輸出關鍵幀
```

### 6. Ground Truth 擷取（選用，僅開發評測時需要）

三種擷取方式，寫出的錄影檔格式**完全相同**，下游不需要知道資料是怎麼來的：

| 子命令 | 機制 | 裝憑證？ | 導向？ | 能錄 Steam 版？ |
|---|---|---|---|---|
| **`cdp`**（預設） | 瀏覽器除錯介面 | 不用 | 不用 | 否（僅網頁版） |
| `proxy` | MITM + 系統代理 | 要 | 要自己設 | 是（若客戶端遵循） |
| `local` | MITM + 行程重導 | 要 | 內建 | 是 |

```bash
python tools/fetch_liqi.py                                  # 更新協定定義

# 推薦：零前置設定，開一個受控 Chromium 打網頁版
python tools/gt.py cdp --out data/recordings/g1/ws.jsonl

# 連到自己已開的 Chrome（需 --remote-debugging-port=9222，保留登入狀態）
python tools/gt.py cdp --connect http://localhost:9222 --out data/ws.jsonl

# MITM：行程重導（mitmproxy 內建，不需要 Proxifier）
python tools/gt.py local --spec Jantama_MahjongSoul --out data/ws.jsonl

# 檢視錄影並轉成 MJAI 事件流
python tools/gt.py inspect data/ws.jsonl --actions --mjai-out data/g1.mjai.jsonl
```

**為什麼預設用 CDP** —— 它不是中間人攔截。瀏覽器自己完成 TLS 交握、自己解密，
我們只是透過瀏覽器原生的除錯介面讀取**已經解密好**的 frame。因此不需要安裝憑證、
不需要流量導向（Proxifier、系統代理、macOS 系統延伸模組授權全都免）、也不怕客戶端
做憑證固定。而且它不介入連線本身——工具關掉最多就是沒在錄，不會像流量重導那樣
可能在錄製中途把連線弄斷。

`cdp` 首次使用需下載瀏覽器：`playwright install chromium`。

**MITM 兩種模式的注意事項**：都需要安裝並信任 mitmproxy 的 CA 憑證
（啟動後瀏覽 http://mitm.it）。`local` 模式的行程重導是 mitmproxy 內建的
（macOS 用 Network Extension、Windows 用 WinDivert），等同 Proxifier 的功能但
不需額外安裝——不過首次啟用時作業系統會要求手動核准系統延伸模組，這一步
無法自動化。憑證自動安裝在 mitmproxy 12.2.3 尚未接上（`mitmproxy_rs` 裡有
`add_cert()` 但 Python 端沒有任何地方呼叫它）。

**仍未驗證**：雀魂是 Unity 原生客戶端，它的網路堆疊會不會查詢系統憑證信任清單
目前沒有答案——就算導向設定正確，仍可能因憑證不被信任而連不上。這是 MITM 兩條路
共同的未知數，也是預設走 CDP 的另一個理由。

---

## 開發里程碑

| # | 內容 | 狀態 |
|---|---|---|
| M0 | 專案骨架、設定載入、日誌 | ✅ 完成 |
| M1 | 擷取層雙平台 + 校正 | ✅ macOS 完成並實機驗證;Windows 待驗證 |
| **M2** | **GT 擷取 + recorder** | ✅ CDP 擷取與完整 liqi→MJAI 管線已用真實對局驗證;MITM 兩模式已實作但未實測 |
| M3 | vision 靜態牌面辨識 | |
| M4 | tracker 事件流 + validator + recovery | |
| M5 | eval 準確率報告 | |
| M6 | engine 子程序 + Mortal 接入 | |
| M7 | UI 側邊視窗 | |
| M8 | Overlay 模式 | |
| M9 | 三方比較評測 | |

**M2 刻意排在 M3 之前。** 先錄下「畫面 ↔ 完美 GT」的成對資料集，
之後所有 CV 開發都能離線跑、自動評分、寫迴歸測試，不必每次開遊戲手動比對。
這一步省下的時間會遠超過它自己的成本。

### 三方比較（M9）

`engine/multiplex.py` 將同一條 MJAI 事件流廣播給 N 個引擎，收集各自的建議：

| 引擎 | 說明 |
|---|---|
| A | Mortal + Akagi 公開權重 |
| B | Mortal + 自行訓練權重 |
| C | 規則式 baseline（向聽最小化，使用 `mahjong` 套件） |

輸出：並排建議顯示、一致率統計、整份牌譜的 agreement matrix。

---

## 候選延伸：打法風格微調（尚未動工）

**狀態：構想已完成可行性調查，實作刻意延後到 CV 主線（M3–M5）之後。**

這個功能與 CV、tracker、UI **完全解耦** —— 它的產出只是一份 `.pth` 權重，
放進 `models/` 就能被 `engine/multiplex.py` 當成第四個引擎載入。
因此它可以在 UI 做好之前、之後、甚至最後才補上，不會卡住任何其他里程碑。

### 動機

目前檯面上的麻將 AI（Mortal、NAGA、Suphx）打法高度相似，都朝同一個目標最佳化。
若能從高段玩家牌譜中**分群出不同打法風格**，再各自微調出一個模型，
就能得到「同一個引擎、不同風格」的多個對照組 —— 這是 Akagi 這類「掛上公開權重」
的工具做不到的事，也是本專題相對於它的差異化賣點。

### 可行性調查結論（2026-07-28）

**微調本身不需要修改 Mortal 任何一行程式碼。**

| 項目 | 結論 |
|---|---|
| 續訓 | [`mortal/train.py`](https://github.com/Equim-chan/Mortal/blob/main/mortal/train.py) 會在 `state_file` 存在時載入 `mortal` / `current_dqn` / `aux_net`（以及 optimizer、scheduler、scaler）並還原 `steps`。把它指向公開權重就是微調 |
| **按玩家過濾資料** | `[dataset] player_names_files` 是**一級功能**：train.py 讀入名單 → 只保留有這些玩家參與的對局 → 把 `player_names` 傳給 `GameplayLoader`，使**只有這些玩家的決策成為訓練樣本**。風格微調需要的正好是這個 |
| 評測 | `one_vs_three.py` + `[1v3]` 設定：challenger 對三個 champion，輸出各名次分布與 `avg_rank`、`avg_pt`，並把對局寫成 **mjai 格式 log** |
| 風格指標 | 上述 log 正是 M2 已在解析的格式 —— 副露率、立直率、放銃率可直接用既有工具算 |
| 版本相容 | [mortal-298k](https://huggingface.co/VoidShine/mortal-298k) 的 `config.toml` 是 `version = 4`、resnet `conv_channels = 192` / `num_blocks = 40`，與現行 repo 的 `config.example.toml` 一致 |

**一個對應「風格 vs 強度」的現成旋鈕** —— 離線階段不是純模仿學習，是
**CQL（Conservative Q-Learning，離線強化學習）**。`[cql] min_q_weight` 控制策略被
拉回資料分布的程度：調高 = 更貼近該群玩家的實際打法（風格更明顯），
調低 = 更自由地往最優化跑。掃過一輪就能畫出**「風格強度 vs 平均順位」的取捨曲線**，
比單一個微調結果更有論文價值。

### 唯一的前置關卡

**沒有公開的 GRP 權重，而離線訓練需要它。**
`dataloader.py` 會建立 `GRP(**config['grp']['network'])` 並載入 `config['grp']['state_file']`
來計算每局的 reward。但 HuggingFace 上的 298k repo 只有 `mortal_298k.pth` 與 `config.toml`，
Mortal 的 GitHub 也**沒有任何 release** —— 兩處都沒有 `grp.pth`。

所以必須先用 `train_grp.py` 自己訓一個。好消息是它很小
（GRU，`hidden_size = 64`、`num_layers = 2`），成本遠低於主模型。**這是動手的第一步。**

### 預期流程

1. 天鳳牌譜 → mjai `.json.gz`（每行一個 JSON，第一行 `start_game` 含 `names`）
2. 計算每位玩家的統計特徵 → 分群成數個風格
3. 每群輸出一份玩家名單檔 → 填入 `player_names_files`
4. **先訓 GRP**（唯一前置關卡）
5. 從 `mortal_298k.pth` 起步，每群各微調一個模型（低學習率）
6. `one_vs_three` 對戰 → 用既有 MJAI 工具算風格指標與平均順位
7. 掃 `min_q_weight` → 風格強度 vs 順位的取捨曲線

**建議的第一個驗證步驟**：先把 libriichi 建起來、載入 298k 權重跑一次推論。
這一步同時驗證建置、GPU 環境、權重相容性三件事，失敗的話也是最早、最便宜的失敗點。

### 硬體與環境

- Python **3.12**（`environment.yml`）
- 建置：`cargo build -p libriichi --lib --release`，再把產物複製成
  `mortal/libriichi.pyd`（Windows）或 `libriichi.so`（Linux）—— **不需要 maturin**
- Windows 上的 Rust 建置需要 MSYS2
- 可用硬體：AMD RX 9070 16G（**ROCm 已正式支援 Windows**，已驗證）／ MacBook M5 Pro 24G
- 授權：Mortal 為 AGPL-3.0，微調後的權重與程式屬衍生作品，論文與散布時須註明

---

## 已知技術難點

| # | 問題 | 對策 | 狀態 |
|---|---|---|---|
| 1 | 全螢幕擷取會把自己的 Overlay 拍進去 | 主路徑用**單視窗擷取**：macOS `CGWindowListCreateImage(windowID)`、Windows `PrintWindow` + `PW_RENDERFULLCONTENT`（硬體加速視窗必需）。mss 僅作 fallback | ✅ macOS 已解決（見下方實測） |
| 2 | macOS Retina 邏輯座標 ≠ 像素座標 | ROI 一律存成相對牌桌矩形的 0~1 正規化座標，執行期再乘實際像素尺寸（同時解決視窗縮放問題） | ✅ 已實作 |
| 3 | 視窗擷取含 OS 標題列 / 瀏覽器工具列 | 多輪邊緣剝除：從四邊往內剝「整條顏色一致」的帶狀區域，每輪重新取參考色以處理疊層 | ✅ 已實作，且實測在 Chromium 上也正確（見下方） |
| 4 | 雀魂 3D 傾斜視角 | 手牌接近正面平視；牌河與副露需先 perspective warp 校正，或每個座位用獨立模板集 | M3 |
| 5 | 擷取到動畫中間幀 | debounce：連續 N 幀一致才確認事件 | M4 |
| 6 | 玩家自訂皮膚 / 牌背 | 模板按 skin profile 分目錄，config 可切換 | M3 |

### M1 實機實測（macOS 26.5.2 · M 系列 · Retina 2×）

| 項目 | 結果 |
|---|---|
| `CGWindowListCreateImage` 在 macOS 26 | **仍正常運作**（雖自 macOS 14 起標記 deprecated） |
| 雀魂 Steam 版擷取 | 3024×1740 @ **60 fps**（16.7 ms/frame） |
| **視窗被遮擋時** | **仍可正確擷取** —— 抓的是視窗自身緩衝區。**難點 #1 在 macOS 上根本不會發生，Overlay 可直接疊在遊戲上** |
| 記憶體格式 | BGRA / little-endian / premultiplied-first → `cvtColor(BGRA2BGR)` |
| mss fallback | 只回傳邏輯解析度 1512×870（像素少一半），確定僅作備援 |
| macOS 標題列 | Retina 下固定 56 px，已由邊緣剝除自動排除 |

實測期間修掉的三個真實缺陷：

1. **`cv2.cvtColor` 取代 numpy 跨步切片**做 BGRA→BGR，兩者輸出完全相同但快約 **50 倍**（0.16 ms vs 7.9 ms）。這是熱路徑，直接把擷取幀率從 47 fps 拉到 73 fps。
2. **圓角視窗破壞背景色推測** —— macOS 視窗四角是透明像素，premultiplied 後變純黑，導致亮色標題列的視窗被誤判成「背景是黑的」。背景取樣改為排除四角。
3. **視窗比對的假陽性** —— VS Code 開著檔名含「雀魂」的檔案時，標題命中且視窗比遊戲還大，「取最大」的啟發式會選錯。改為加權評分：程式名稱 > 標題完全相符 > 標題子字串。

尚未確定、留待 M2 用真實資料集釐清：雀魂在 macOS Steam 版似乎是**撐滿視窗內容區**（量到寬高比 1.806）而非嚴格 16:9 letterbox，因此 `enforce_aspect` 預設關閉 —— 在這種情況下強制裁成 16:9 反而會讓所有 ROI 偏移。

### M1 擷取層在 Chromium 上的實測

GT 改走 CDP 之後，錄畫面的對象從 Steam 版變成 Playwright 開的 Chromium，
因此重新驗證了擷取與校正：**兩者都正常**。校正抓到的牌桌矩形是
`2560x1430@(0,214)`，y 偏移 214 物理像素（= 107 邏輯像素）正確剝掉了分頁列與網址列，
下緣也正確排除黑邊，寬高比 1.7902。M1 的「多輪邊緣剝除」不是寫死剝標題列，
而是從四邊往內剝顏色一致的帶狀區域，所以原生標題列與瀏覽器 chrome 都能處理。

但實測抓到**視窗自動比對會選錯**：VS Code 開著檔名含「雀魂」的檔案時，
它與真正跑著網頁版雀魂的 Chromium **拿到完全相同的分數**——兩者程式名都不符，
都只命中標題子字串——然後同分比面積，較大的 VS Code 就贏了。

修法是加入 `host_patterns`（瀏覽器程式名）作為獨立加分項，
但**只有在標題也命中時才加分**，否則任何瀏覽器視窗都會被誤選。
另外當勝出者只靠標題子字串命中時會發出警告——靜靜錄錯視窗是最糟的失敗方式。

### M2 實測與設計決策

**協定定義從哪來** —— Steam 桌面版是 Unity 原生客戶端，協定沒有以檔案形式散布
（4.6 GB 的 StreamingAssets 全是混淆過的 `.majset`，`Assembly-CSharp.dll` 裡也沒有
protobuf descriptor）。但**網頁版公開提供同一份 `liqi.json`**，兩者協定相同。
[fetch_liqi.py](tools/fetch_liqi.py) 沿 `version.json → resversion → liqi.json` 三段路徑取得
（不能直接猜路徑：資源版本 `v0.11.243.w` 落後客戶端版本 `0.11.252.w`）。
取得的定義有 **1318 個訊息、439 個 RPC 方法、4625 個欄位**，全數成功轉成 Python
protobuf 型別，已隨專案版控在 `assets/proto/`（附 sha256 版本檔，測試會驗證一致性）。

**錄製格式為什麼用 JPEG** —— 直覺上量測準確率該用無損，但實測（3024×1740 真實牌局畫面）：

| 格式 | 大小 | 寫入 | 模板比對峰值 |
|---|---|---|---|
| PNG 級別 1 | 4.10 MB | 127 ms | 1.000000 |
| PNG 級別 9 | 3.22 MB | 7710 ms | 1.000000 |
| **JPEG q95** | **1.00 MB** | **9 ms** | **0.999800** |

PNG 的 127 ms 寫入超過 10 fps 的取樣週期，錄製時會直接掉幀；JPEG q95 對模板比對的
影響只在小數第四位、峰值位置完全相同（平均像素差 1.01）。需要嚴格無損時用
`--format png` 錄一小段對照組即可。

**畫面去重** —— 麻將牌桌大部分時間靜止。實測一場對局約 **80% 的幀沒有變化**，
只存有變化的幀但每幀都記時間戳（時間軸完整、事件才對得齊），45 秒錄影從 378 MB 降到 119 MB。

**擷取與解析分離** —— 擷取端只做一件事：把原始 WebSocket frame 以 base64 寫成
JSONL。liqi 解析全在離線階段。這樣解析程式有 bug 或協定改版時，重跑一次就好，
不必重打一場牌；也讓跑在 mitmproxy 事件迴圈裡的程式碼盡可能不會出錯。

這個分離還帶來一個原本沒預期的好處：**換擷取方式完全不影響下游**。後來加入 CDP
後端時，只需要讓它寫出同樣格式的檔案，解析、MJAI 轉換、評測一行都不用改。
`DumpWriter` 是兩者共用的格式定義，測試裡有一組 parity 測試專門盯住這件事。

**liqi → MJAI 的轉換語意** —— 參考 [Akagi](https://github.com/shinkuan/Akagi)（Apache-2.0）
的 Majsoul bridge 說明文件，以 Python 重新實作，未複製其原始碼（見 [NOTICE](NOTICE)）。
該文件記錄了幾處容易做錯、而且其他實作確實做錯了的地方：

- **配牌**：雀魂發莊家 14 張、其他人 13 張；MJAI 是所有人 13 張再補一個 `tsumo`
- **寶牌翻牌時機**：暗槓是即乗り（`ankan → dora → 嶺上 tsumo`），加槓與大明槓是後乗り（`kan → 嶺上 tsumo → dora → dahai`）
- **`reach_accepted`**：宣言牌安全通過後才記點，插在下一個動作之前；被榮和則立直作廢
- **榮和放銃者**：`HuleInfo` 不帶目標座位，必須追蹤「最後亮牌的人」——只看切牌會讓搶槓、國士搶暗槓、三麻搶北全部判錯
- **`GameRestore` 內夾帶的動作未經 XOR 混淆**，與即時動作不同

同時交叉驗證了先前標為「未驗證」的假設：XOR 金鑰表與線路格式與其實作完全一致。

**「能解析」不等於「解對了」** —— 實測發現 6 位元組的 `ActionDiscardTile` 做 XOR 之後
**照樣能被 protobuf「成功」解析**（所有位元組落進 unknown fields、已知欄位全空，
連 round-trip 都一致）。只靠 try/except 選分支會拿到一個空的動作。判準因此改為
「有沒有解出任何已知欄位」，並依來源（即時 vs 重連）決定先試哪一邊。

**真實流量抓出一個合成測試抓不到的 bug** —— 第一次錄真實對局時,242 個牌局動作
**一個都沒解析出來**,而當時 262 項測試全數通過。原因是 `ActionPrototype.name`
在實際協定裡是**未限定的裸名**（`"ActionDiscardTile"`），不像外層 `Wrapper.name`
是完整名（`.lq.ActionPrototype`）。我照著自己的錯誤假設造測試資料，於是測試只證明了
「實作符合我的理解」，證明不了「我的理解符合現實」。

修正後這場對局的結果：**569 個 frame 100% 解析成功、242 個動作、249 個 MJAI 事件**，
且 **242/242 全部需要 XOR 反混淆** —— 這同時證實了金鑰表與嘗試順序都正確。
節錄已收進 `tests/fixtures/real_game_excerpt.jsonl` 當作永久迴歸測試。

**和了的點數增減加總不是零** —— 是 1000 的倍數。立直棒在 MJAI 記帳模型裡由
`reach_accepted` 扣除、不進 `delta_scores`，但贏家收供托時會算進去，差額恰好是
收走的立直棒數。這是寫測試時先斷言成零、被真實資料打臉才發現的語意。

**每條 WebSocket 連線一個解析器** —— 各連線有獨立的 `msg_id` 序列，共用解析器會讓
請求與回應互相配錯、用錯誤的型別去解。錄影檔因此必須記錄 `flow` 欄位。

---

## 技術選型

- **PySide6 而非 PyQt6** — API 幾乎相同，但 PySide6 是 LGPL、PyQt6 是 GPL/商業雙授權。
  本專案已牽涉 AGPL 的 Mortal，沒有必要再疊一層 GPL。
- **`mahjong` 套件** 負責向聽數與點數計算 —— 同時作為 baseline 引擎與 tracker 的規則校驗依據。
  不自行重寫向聽演算法。
- **主環境不安裝 torch** —— 只存在於 `engines/mortal/`。主程式保持輕量、啟動快。

---

## 授權與使用聲明

- **Mortal** 為 [AGPL-3.0](https://github.com/Equim-chan/Mortal)。本專案透過子程序與 IPC 協定
  與其通訊以維持授權隔離；若日後改為直接連結其程式碼，本專案須同樣採用 AGPL-3.0 發佈。
- 模型權重不隨本 repo 散布，請依 Mortal 專案的規範自行取得。
- **本專案為學術研究用途。** 第三方輔助工具可能違反雀魂使用條款；
  展示與實驗建議使用觀戰模式或牌譜回放進行，勿用於排名對局。
