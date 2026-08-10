# MIA — Mahjong Intelligence Assistant

跨平台（macOS / Windows）的雀魂麻將輔助系統。大學畢業專題。

由**三個互相獨立的功能**組成：

| # | 功能 | 狀態來源 | 現況 |
|---|---|---|---|
| 1 | **手牌辨識 + 向聽計算** | 螢幕擷取 + CV（只需自己的手牌） | 進行中 |
| 2 | **麻將 AI 建議** | WebSocket 封包解析 → MJAI → Mortal | 事件流已完成 |
| 3 | **打法風格微調權重** | 離線訓練（Mortal fine-tuning） | 可行性已調查，尚未動工 |

三者刻意解耦：功能 1 只吃畫面、功能 2 只吃封包、功能 3 產出一份 `.pth`。
任何一個做不完都不會擋住其他兩個。

> **這個結構是專案中期調整過的。** 原本的設計是用 CV 辨識整個牌桌並重建完整
> 對局狀態，封包只當 Ground Truth。調整的原因、被放棄的東西、以及已經量測完成
> 但不再使用的座標，完整記在 **[docs/decisions.md](docs/decisions.md)**。

> **接手這個專案？** 先看 **[docs/handoff.md](docs/handoff.md)** —— 現況、
> 跑法、已經決定過不要再討論的事、以及踩過的坑，一份講完。

---

## 專題定位

賣點是**功能 3**。

功能 2 的做法（封包攔截 + 掛上 Mortal 公開權重）與既有的
[Akagi](https://github.com/shinkuan/Akagi) 同級，是基礎能力而不是差異。
真正的差異在於：目前檯面上的麻將 AI（Mortal、NAGA、Suphx）打法高度相似，
都朝同一個目標最佳化。本專題嘗試從高段玩家牌譜中**分群出不同打法風格**，
各自微調出一個模型，得到「同一個引擎、不同風格」的多個對照組。

功能 1 的價值則在於它**不碰遊戲連線** —— 純看畫面就能給向聽數與進張建議，
是一個不需要封包、也不介入網路的輕量模式。

---

## 架構

### 三進程模型

```
┌───────────────────────────────────────────────┐
│  主程式   (Python 3.14 · .venv)                │
│  capture → vision → ui        (功能 1)         │
│  gt(封包) → mjai → engine → ui  (功能 2)        │
└────────┬───────────────────────────┬──────────┘
         │ MJAI JSON-lines / stdio   │ JSON-lines
         ▼                           ▼
┌──────────────────────┐   ┌──────────────────────────┐
│  AI Engine × N       │   │  封包擷取(二選一)         │
│  (Python 3.12)       │   │  CDP / mitmproxy          │
│  Mortal + libriichi  │   │  寫 .jsonl 錄影檔          │
└──────────────────────┘   └──────────────────────────┘
```

**為什麼 AI 引擎必須是獨立子程序：**

1. `libriichi` 不在 PyPI（需從 Mortal repo 自行編譯），`mjai` 在 PyPI 上只有
   `cp312` wheel —— 兩者都裝不進主程式的 Python 3.14 環境。
2. torch 載入慢、崩潰時不該拖垮 UI。
3. 多模型比較 = 同時啟動多個引擎子程序，餵同一條事件流。
4. 授權邊界清楚：Mortal 是 AGPL-3.0，透過子程序 + 明確定義的 IPC 協定通訊，
   一般視為獨立程式。**這一點現在只是附帶好處** —— 本專案已主動採用 AGPL-3.0
   （見[授權與使用聲明](#授權與使用聲明)），架構是靠理由 1～3 站住的。

MJAI 本來就是 JSON-lines over stdio 的協定，子程序化是它的原生形式，不是妥協。

### 兩條資料流

```
功能 1   Frame ──read_hand──▶ 牌框 ──classify──▶ 牌名 ──analyse──▶ 向聽 / 進張 / 該切哪張
         (ndarray)   固定槽位      模板比對       雀魂記法      mahjong 套件

功能 2   WebSocket frame ──liqi──▶ MjaiEvent ──▶ engine × N ──▶ Advice
         (base64 .jsonl)           (JSON)
```

即時模式下兩條路各跑一條執行緒，交會點只有一個郵箱：

```
擷取執行緒(15 fps)──┐
                      ├─▶ UpdateBus ──▶ pump()(UI 執行緒)──▶ ViewModel ──▶ Panel
封包執行緒(事件驅動)─┘                                                  └─▶ Overlay
```

`pump()` 是唯一碰 `ViewModel` 的地方，而它只在 UI 執行緒上跑 —— Qt 的 widget
只能在建立它的執行緒上動，而 `ViewModel` 本身也不是執行緒安全的。郵箱**每個
slot 只留最新一筆**：主執行緒卡住的時候要掉幀，不要恢復後把積壓的舊局面
補播一遍。細節見 [docs/decisions.md](docs/decisions.md) 第十節。

兩種 HUD 共用同一個 `ViewModel`，不重複實作顯示邏輯。

### 目錄

```
src/mia/
├── config/        設定模型 (pydantic)、YAML 載入、skin profile
├── capture/       平台擷取抽象:macos(Quartz) / windows(PrintWindow) / mss fallback
├── calibration/   牌桌矩形偵測、DPI 與 Retina 縮放、ROI 正規化
├── vision/        純 CV,無狀態:roi + tiles/(手牌定位 hand + 牌面分類 classify)
├── analysis/      向聽數、進張、打牌建議(mahjong 套件)
├── mjai/          events / tiles(牌表示轉換)
├── engine/        base / subprocess_engine / mortal / dummy / multiplex
├── live/          即時模式:bus / vision / packets / runtime / source
├── ui/            viewmodel + state(記住的位置)+ switchboard + panel/ overlay/ widgets/
├── groundtruth/   cdp / capture_addon / dump / liqi / schema / to_mjai
├── eval/          align / metrics / report
├── recorder/      錄製資料集、離線回放
└── utils/

engines/mortal/    Mortal 子程序的獨立環境 (Python 3.12)
tools/             開發用 CLI:capture_probe / roi_annotate / roi_union / record / gt / fetch_*
assets/tiles/      牌面模板圖(37 張/皮膚),由 tools/fetch_tiles.py 產生
config/            ROI 定義、skin profile YAML
data/              錄製資料集、ui_state.json(記住的 Overlay 位置)(gitignore)
models/            權重 .pth (gitignore)
docs/decisions.md  方向轉折的完整記錄
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

完整步驟見 [`engines/mortal/requirements.txt`](engines/mortal/requirements.txt)。
需要 Rust 與 Python 3.12，自行從 [Mortal repo](https://github.com/Equim-chan/Mortal)
編譯 `libriichi`（`cargo build -p libriichi --lib --release`，**不需要 maturin**），
權重 `.pth` 放到 `models/`（已 gitignore）。

> **macOS 注意**：官方文件只寫了 Linux 與 Windows。在 macOS 上產物是
> `libriichi.dylib`，但 CPython 只認 `.so` —— 要改名複製成 `mortal/libriichi.so`，
> 否則 ImportError。已在 Apple Silicon 實測通過（編譯 32.5 秒，推論 14 ms/決策）。

### 3. 平台權限

- **macOS**：首次執行需在「系統設定 → 隱私權與安全性 → 螢幕與系統錄製」勾選
  執行本程式的應用程式（從終端機跑就是終端機 App，從 VS Code 跑就是 VS Code）。
  **授權後必須完全結束並重新啟動該應用程式**，權限才會生效。
  未授權時的徵兆是：視窗列得出來，但標題全是空的，且擷取回傳 None。
- **Windows**：程式啟動時會設定 `PER_MONITOR_AWARE_V2` DPI awareness，
  否則座標會被系統縮放偷改。

### 4. 取得牌面模板

```bash
python tools/fetch_tiles.py                       # 預設牌面皮膚
python tools/fetch_tiles.py --list-skins          # 看有哪些皮膚
python tools/fetch_tiles.py --skin mjpface_25summer
```

從雀魂官方資源取得牌面圖集並切成 37 張模板（34 種牌 + 3 種赤寶牌），
存進 `assets/tiles/<皮膚>/`。**遊戲裡換了牌面皮膚就要重跑一次**（牌**背**
皮膚不影響——手牌定位那一層已經不看顏色了）。

### 5. 驗證安裝

```bash
python tools/capture_probe.py --list                     # 看得到哪些視窗
python tools/capture_probe.py --out shot.png             # 抓一張雀魂畫面
python tools/capture_probe.py --calibrate --out cal.png  # 校正並輸出標註圖
python tools/capture_probe.py --bench 60                 # 測擷取幀率
```

`--calibrate` 產生的標註圖中，綠框是偵測到的牌桌矩形、橘線是三分格線。
請目視確認綠框有貼齊遊戲畫布——所有 ROI 都以這個矩形為基準，它錯了全部都會錯。
若自動偵測不準，在 `config/default.local.yaml` 以 `manual_table_rect` 手動指定。

`--calibrate` 會**連續取樣多幀**再回報，不是抓一張就算。單幀校正是啟發式，
會被和了動畫、角色立繪、載入畫面干擾 —— 實測同一場錄影的 216 幀產生了
21 種不同的 `table_rect`，其中 16% 明顯錯誤。詳見
[docs/decisions.md](docs/decisions.md) 第二節。

### 6. 錄製資料集

```bash
python tools/record.py --duration 60          # 錄 60 秒
python tools/record.py                        # 錄到 Ctrl-C
python tools/record.py --inspect data/recordings/<id>    # 檢視
python tools/record.py --export data/recordings/<id> --export-dir out/  # 輸出關鍵幀
```

### 7. 封包擷取（功能 2 的狀態來源）

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
| M2 | 封包擷取 → MJAI 事件流 + recorder | ✅ 已用真實對局驗證;MITM 兩模式已實作但未實測 |
| M3 | **功能 1**:手牌 CV | ✅ 校正穩定化、手牌定位、牌面分類皆完成 |
| M3-1b | 準確率評測工具鏈 | ✅ 完成 —— 素材已錄、數字已量,**每張牌正確 97.1%、整手正確 88.0%**,見下方「M3-1b 實測」 |
| M4 | **功能 1**:向聽 / 進張計算 | ✅ 完成 —— `analysis/shanten.py`,整條管線已跑通 |
| M5 | **功能 2**:engine 子程序 + Mortal 接入 | ✅ 完成 —— 已用真實對局驗證,見下方「M5 實測」 |
| M6 | UI 側邊視窗 | ✅ 完成 —— 左側功能列 + 牌面圖片 + Q 值長條 |
| M6-1 | **即時資料連接層** | ✅ 完成 —— 封包這條路已實機驗證;兩條路同時跑尚未一起驗過 |
| M7 | Overlay 模式 | ✅ 完成 —— 疊在遊戲上的精簡 HUD,可拖曳 / 鎖定穿透;**尚未實機疊在對局上看過** |
| M8 | **功能 3**:風格微調(見下節) | |

### M3 的三個子項

1. ~~**修校正不穩**~~ —— ✅ 已完成（`calibration/stable.py`）。多幀取樣 →
   丟掉離譜候選 → 逐分量中位數。用實測分布做蒙地卡羅，「誤差 > 1%」的機率
   從 **13.4% 降到 0%**，最差情況從 287 px 降到 11 px。
   尚未對真實錄影端對端複驗（原始錄影已刪除）。
2. ~~**擺脫可自訂牌背的色相依賴**~~ —— ✅ 已完成（`vision/tiles/hand.py`）。
   改用**固定槽位模型** `x_k = origin + k*pitch`（實測殘差 < 1 px），佔用以
   標準差判斷，完全不看顏色。原本要做的「開局就地量測色相」整項取消。
3. ~~**牌面模板庫 + 分類器**~~ —— ✅ 已完成（`vision/tiles/classify.py`）。
   模板改從**雀魂官方資源**取（`tools/fetch_tiles.py`），37 張已經標好、
   不需要人工標註。24 張目視確認過答案的牌全數正確，最低分 0.631、
   最小差距 0.084。真實對局上的 per-tile 準確率見下方「M3-1b 實測」。

---

## 功能 3：打法風格微調

**狀態：可行性調查完成，實作排在最後。** 產出只是一份 `.pth`，放進 `models/`
就能被 `engine/multiplex.py` 當成額外的引擎載入 —— 與 CV、UI 完全解耦，
什麼時候補都行。

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
- 建置：`cargo build -p libriichi --lib --release`（**不需要 maturin**）
- Windows 上的 Rust 建置需要 MSYS2
- 可用硬體：AMD RX 9070 16G（**ROCm 已正式支援 Windows**，已驗證）／ MacBook M5 Pro 24G
- 授權：Mortal 為 AGPL-3.0，本專案亦同。微調權重是否構成衍生作品法律上無定論，
  採保守立場處理，詳見[授權與使用聲明](#授權與使用聲明)

---

## 已知技術難點

| # | 問題 | 對策 | 狀態 |
|---|---|---|---|
| 1 | 全螢幕擷取會把自己的 Overlay 拍進去 | 主路徑用**單視窗擷取**：macOS `CGWindowListCreateImage(windowID)`、Windows `PrintWindow` + `PW_RENDERFULLCONTENT` | ✅ macOS 已解決（見下方實測） |
| 2 | macOS Retina 邏輯座標 ≠ 像素座標 | ROI 一律存成相對牌桌矩形的 0~1 正規化座標，執行期再乘實際像素尺寸 | ✅ 已實作 |
| 3 | 視窗擷取含 OS 標題列 / 瀏覽器工具列 | 多輪邊緣剝除：從四邊往內剝「整條顏色一致」的帶狀區域 | ✅ 已實作，Chromium 上也正確 |
| 4 | 單幀校正是啟發式，會被動畫與立繪干擾（216 幀產生 21 種 `table_rect`，16% 明顯錯誤） | `StableCalibrator`：多幀取樣 → 丟掉離譜候選 → 逐分量中位數 | ✅ 已修，尚未對真實錄影複驗 |
| 5 | 玩家自訂牌背改變切牌的色相基準 | 改用固定槽位 + 標準差判斷佔用，不看顏色 | ✅ 已解決，色相依賴整個移除 |
| 6 | 雀魂 3D 傾斜視角 | 手牌接近正面平視，可直接處理 | ✅ 牌河已移出範圍，不再是問題 |

### M1 實機實測（macOS 26.5.2 · M 系列 · Retina 2×）

| 項目 | 結果 |
|---|---|
| `CGWindowListCreateImage` 在 macOS 26 | **仍正常運作**（雖自 macOS 14 起標記 deprecated） |
| 雀魂 Steam 版擷取 | 3024×1740 @ **60 fps**（16.7 ms/frame） |
| **視窗被遮擋時** | **仍可正確擷取** —— 抓的是視窗自身緩衝區。**難點 #1 在 macOS 上根本不會發生，Overlay 可直接疊在遊戲上** |
| mss fallback | 只回傳邏輯解析度 1512×870（像素少一半），確定僅作備援 |

實測期間修掉的三個真實缺陷詳見 [docs/decisions.md](docs/decisions.md) 第四節。

### M2 實測與設計決策

**協定定義從哪來** —— Steam 桌面版是 Unity 原生客戶端，協定沒有以檔案形式散布
（4.6 GB 的 StreamingAssets 全是混淆過的 `.majset`）。但**網頁版公開提供同一份
`liqi.json`**，兩者協定相同。[fetch_liqi.py](tools/fetch_liqi.py) 沿
`version.json → resversion → liqi.json` 三段路徑取得。取得的定義有
**1318 個訊息、439 個 RPC 方法、4625 個欄位**，全數成功轉成 Python protobuf 型別，
已隨專案版控在 `assets/proto/`（附 sha256 版本檔，測試會驗證一致性）。

**擷取與解析分離** —— 擷取端只做一件事：把原始 WebSocket frame 以 base64 寫成
JSONL。liqi 解析全在離線階段。這樣解析程式有 bug 或協定改版時，重跑一次就好，
不必重打一場牌。這個分離還帶來一個原本沒預期的好處：**換擷取方式完全不影響下游**
—— 後來加入 CDP 後端時，只需要讓它寫出同樣格式的檔案，解析、MJAI 轉換一行都不用改。

**liqi → MJAI 的轉換語意** —— 參考 [Akagi](https://github.com/shinkuan/Akagi)（Apache-2.0）
的 Majsoul bridge 說明文件，以 Python 重新實作，未複製其原始碼（見 [NOTICE](NOTICE)）。
幾處容易做錯的地方：

- **配牌**：雀魂發莊家 14 張、其他人 13 張；MJAI 是所有人 13 張再補一個 `tsumo`
- **寶牌翻牌時機**：暗槓是即乗り（`ankan → dora → 嶺上 tsumo`），加槓與大明槓是後乗り
- **`reach_accepted`**：宣言牌安全通過後才記點，插在下一個動作之前；被榮和則立直作廢
- **榮和放銃者**：`HuleInfo` 不帶目標座位，必須追蹤「最後亮牌的人」
- **`GameRestore` 內夾帶的動作未經 XOR 混淆**，與即時動作不同

真實對局驗證結果：**569 個 frame 100% 解析成功、242 個動作、249 個 MJAI 事件**，
且 **242/242 全部需要 XOR 反混淆**。節錄已收進
`tests/fixtures/real_game_excerpt.jsonl` 當作永久迴歸測試。
更多實測發現（含一個合成測試抓不到的 bug）見 [docs/decisions.md](docs/decisions.md) 第四節。

### M5 實測：引擎子程序與 Mortal 接入

**協定是嚴格的一問一答。** MJAI 原本的形式是「有動作才輸出」，但那樣父程序分不出
「引擎不打算動作」與「引擎還在算」，只能等滿逾時 —— 一場 250 手就是好幾分鐘的純等待。
子程序端不動作時補一行 `{"type":"none"}`，這個問題就完全消失。啟動時另外要求一行
`hello`，用來區分「還在載入 130 MB 權重」與「已經死了」。

**子程序端沒有沿用上游的 `mortal.py`。** 它只在有動作時輸出，而它的 `review_mode`
雖然會補 `none`，卻在 stdin 關閉後接著跑 GRP 分析 —— 本專案還沒有 GRP 權重，每次
關閉都會以例外收場。`engines/mortal/bot.py` 是一層薄殼，只多做握手、補 `none`、
把例外轉成一行 JSON；推論完全交給上游的 `MortalEngine` 與 `libriichi.mjai.Bot`。
它也**不 import 上游的 `prelude`** —— 那會連帶要求安裝 tensorboard，那是訓練才需要的。

**重啟不等於復原。** MJAI 引擎有狀態，它自己從事件流重建局面，所以崩潰後起一個新
行程只會拿到一個對這局一無所知的引擎，而且**不會報錯**。`SubprocessEngine` 預設
會重送先前的事件讓它追上進度。

用 [tools/advise.py](tools/advise.py) 把 M2 錄下的真實對局重播給引擎：

| 素材 | 決策點 | `mortal_298k` | 規則式 baseline | 兩者彼此 |
|---|---|---|---|---|
| 完整東風戰（550 事件） | 73 | **66%** | 42% | 53% |
| 早期節錄（249 事件） | 33 | **76%** | 39% | 42% |

延遲：Mortal 10~23 ms/決策，baseline < 1 ms。

兩個引擎彼此只有一半左右一致 —— 這個差距正是功能 3 要調的東西。
**一致率不是準確率**：人打錯的時候引擎不跟著錯，這個數字反而會下降，
它衡量的是「像不像這個人」。

```bash
python tools/advise.py data/gt/ws.jsonl --mortal models/mortal_298k.pth
```

### 即時模式：真的接上遊戲

```bash
# 按左下角「開始遊戲」開瀏覽器，登入後撥開關就開始給建議
python tools/ui.py --live --mortal models/mortal_298k.pth

# 已經有另一個 gt.py 在錄了（或用 MITM 錄 Steam 版），只跟著那個檔案走
python tools/ui.py --live --tail data/recordings/now/ws.jsonl

# 只要 AI 建議，不跑畫面辨識（不必授權螢幕錄製）
python tools/ui.py --live --no-packets   # ← 反過來：只要畫面辨識，不接封包
```

**瀏覽器不會自己開**，要按左下角的「開始遊戲」。每按一次是新的一場，寫到新的
錄影檔 —— 錄影檔是 append 模式寫的，而封包這條路是從檔頭讀的，兩場疊在同一個
檔案裡的話引擎會拿著一個已經結束的牌局給建議**而且不會報錯**。關掉瀏覽器之後
再按一次就能開下一場，不必重開 MIA。

**兩個功能預設都不執行**，各自由所在頁面右上角的開關控制（iOS 風格，Apple 的
system green）。打開 AI 建議會開一個載著 130MB 權重的子程序、打開畫面辨識會
開始持續擷取螢幕 —— 這兩件事都該是明確的動作，不是打開視窗的副作用。

中途才打開也沒問題：錄影檔從一開始就在寫，封包執行緒從檔案開頭讀，座位與
寶牌都補得回來（實測追完整場約 2 秒）。擷取子程序**不受開關影響**，因為綁上去
的話關掉再打開會殺掉瀏覽器、整場對局就沒了。

`--live` 會開一個 `tools/gt.py cdp` 子程序抓封包，同時擷取遊戲視窗跑 CV。
**封包錄影是副產品**：即時建議與離線準確率評測用的是同一份 `ws.jsonl`，
不必為了產生資料集再打一場。

擷取仍然是子程序而不是塞進主程式，理由與 `gt.py` 當初對 mitmdump 的決定相同
—— Playwright 與 mitmproxy 各自帶著自己的事件迴圈，而主程式這邊還有一個
Qt 事件迴圈要顧。代價是多一次落地與讀回，換到的是「三種擷取方式都能用」
與「瀏覽器崩掉不會拖垮 UI」。

**引擎在對局開始前就先起好。** 載 130MB 權重實測 0.5~0.6 秒 —— 不長，但發牌
那一刻才開始載仍然來不及（第一次摸牌就在幾秒之內）。
所以座位不能靠啟動參數寫死 —— `bot.py` 改成每收到 `start_game` 就照它的 `id`
重建 `libriichi` 的 `Bot`（權重原封不動留著，重建幾乎免費）。

即時路徑的實機驗證（把整場東風戰當成正在寫入的錄影檔跟著跑）：

| 項目 | 結果 |
|---|---|
| 事件 / 決策點 | 550 / 73 —— 與離線重播完全相同 |
| 座位辨識 | 刻意用 `--seat 0` 啟動，正確跟著事件流改成 2 |
| 全場耗時 | 2.0 秒（含 Mortal 每一手的推論） |
| 郵箱合併 | 投遞 210 筆 → UI 只收到 **12 次**通知 |
| 半截行處理 | 1398 行、**0 個壞行**（刻意在第 400 行切一次） |

**尚未一起驗過的是「畫面與封包同時在真實對局上跑」** —— 兩條路各自都在真實
素材上驗過了，但需要一次實機對局才能驗它們一起跑。

操作步驟、狀態列訊息對照表、卡住時怎麼查 —— 見 **[docs/live.md](docs/live.md)**。

### 準確率評測：拿封包當標準答案

純 CV 沒辦法宣稱 100%，但**可以讓錯誤變成一個可量測、可追蹤的數字**，
並指出錯在哪幾種牌上。混淆對照表通常比總體準確率有用得多 ——
「9s 被認成 4s」是可以修的，「準確率 93%」不是。

同時錄下的畫面與封包靠**牆上時鐘**配對（session manifest 存了第一幀的
`time.time()`，錄影檔每一行也有 `wall`）。

**只取穩定區間。** 封包在動作發生時就到了，畫面要等動畫演完才變 ——
轉場的那半秒裡兩者必然不一致，那不是 CV 認錯。所以一個狀態前後都安靜夠久，
落在中間的幀才拿來評分。代價是可用的幀變少，好處是剩下的每一幀都有明確的
標準答案。反過來用一個容忍度去湊，準確率就變成由容忍度決定的數字了。

```bash
# 先確認兩份錄影對得上（不跑 CV，幾秒就有結果）
python tools/evaluate.py data/recordings/<id> data/gt/ws.jsonl --dry-run
python tools/evaluate.py data/recordings/<id> data/gt/ws.jsonl
```

錄製步驟見 [docs/recording.md](docs/recording.md)。

### M3-1b 實測（2026-08-01 錄、2026-08-03 定案、2026-08-10 複測）

12 局 / 2 場完整東風戰，2395 個封包 frame（100% 解析）配 12599 幀畫面，
牌桌 2536×1430。穩定秒數 1.0s，配到 **4327 幀**有明確標準答案：

| | 2026-08-03 | 2026-08-10 |
|---|---|---|
| 張數正確 | 88.4% | 88.4% |
| **每張牌正確** | 95.8% | **97.1%** |
| **整手正確** | 74.3% | **88.0%** |
| 赤寶牌召回 | **100.0%** | **100.0%** |

兩欄是**同一批素材、同一組參數**，差別只有 8/10 修掉的「滑鼠停在待選牌上會
把牌抬起來」（見下）。整手正確跳 13.7 個百分點而張數完全沒動，正好對上那個
bug 的形狀：它一次只弄壞**一張**牌，而整手正確是 13 張裡錯一張就算錯。

最常認錯的從 `E → 5m`、`P → 4p` 變成 `9m → P`、`8m → P` —— 剩下的錯誤集中在
**把移動中的牌認成白板**。白板的牌面就是一片空白，摸打動畫中途被拍到的牌
（半透明、位移中）與它最像。這是下一個值得動的地方。

#### 滑鼠抬牌（2026-08-10）

滑鼠停在待選牌上時，雀魂會把那張牌往上抬。實測抬 **65 px / 牌高 205 px
（32%）**，88 個樣本全部落在 65±5。手牌 ROI 是貼著牌量的，抬起來的那張有
三分之一跑到框外，而 `classify` 的平移搜尋範圍只有牌高的 16%
（`FACE_RATIO` 剩下的餘裕）—— 搜不到，那張牌就認錯，整手的向聽數跟著錯。

**而使用者滑到某張牌的時機，正是他在看建議的時機。**

修法是往 ROI 上方多切 `HOVER_HEADROOM`（0.5 個牌高）給它搜；同一張牌在抬起
狀態下的相關係數從搜不到回到 0.99。代價是比對成本 +50%
（434 → 653 ms / 手，2560 牌桌）。

跑這一輪修掉三個一直沒被發現的東西：

1. **`tools/evaluate.py` 的 `Roi.pixels` 根本不存在** —— 工具鏈的 CV 那一段
   從來沒有真的執行過。`_own_hand_roi` 掛著 `type: ignore[no-untyped-def]`,
   回傳值變成 `Any`，mypy 就看不見了。沒有型別的地方就是沒有人在看的地方。
2. **時間軸不會說「現在沒有手牌」** —— 局末到下一局那十幾秒,`bisect` 一律
   回答上一局的最後一手,每一幀都被判成 CV 認錯。修掉之後同一批素材從
   68% 升到 81.3%。
3. **穩定秒數是量出來的,不是估的** —— 掃 0.4~2.0 秒,張數正確率在 **1.0 秒
   附近平掉**,所以雀魂的摸打與鳴牌動畫實際約 1 秒(原本估 0.3~0.4)。

第二段素材(視窗 900×593)一開始只有 29.1%,查出來是**牌桌校正鎖進了錯的
矩形**,不是正規化 ROI 座標跨解析度失效 —— 用正確矩形重跑是 96.0%。
這件事直接催生了固定畫布尺寸功能,見 `docs/decisions.md` 第十六節。

---

## 技術選型

- **PySide6 而非 PyQt6** — API 幾乎相同，但 PySide6 是 LGPL、PyQt6 是 GPL/商業雙授權。
  當初的理由是「避免再疊一層 GPL」；本專案自己採用 AGPL-3.0 之後，這個理由已經不成立
  （AGPL 專案用 PyQt6 完全沒問題）。**保留 PySide6 是因為換掉沒有好處**：
  LGPL 對呼叫端的約束比 GPL 寬鬆，日後若有人想以其他授權 fork 本專案的 UI 層，
  PySide6 少擋一道；且 6.11 有 abi3 wheel、支援 3.14。
- **`mahjong` 套件** 負責向聽數與點數計算 —— 功能 1 的計算核心。不自行重寫向聽演算法。
- **主環境不安裝 torch** —— 只存在於 `engines/mortal/`。主程式保持輕量、啟動快。

---

## 授權與使用聲明

本專案採 **[AGPL-3.0-or-later](LICENSE)**（GNU Affero General Public License v3.0 或更新版本）。

### 為什麼是 AGPL

這是**主動選擇**，不是被動繼承。

專案架構上已經與 Mortal 保持授權隔離（獨立子程序 + MJAI stdio 協定，一般視為兩個獨立程式），
理論上主程式可以自選任何授權。但「子程序算不算獨立作品」這件事在法律上**沒有判例支撐**，
只有 FSF 的解釋。既然本專案本來就要在 GitHub 上公開、也不打算商業化，
主動採用與上游相同的 AGPL-3.0 就讓那個論證從「風險評估」變成「不需要成立」。

順帶說明：**子程序架構的存廢與授權無關。**`libriichi` 不在 PyPI、`mjai` 只有 cp312 wheel，
兩者都裝不進主程式的 Python 3.14 環境 —— 這個技術限制本身就足以決定架構
（見上方「為什麼 AI 引擎必須是獨立子程序」的理由 1～3）。

### 這代表什麼

- 任何人都可以取得、修改、再散布本專案，但**衍生作品必須同樣以 AGPL-3.0 釋出並公開原始碼**。
- AGPL 第 13 條：若日後把本專案**做成網路服務**，透過網路使用它的人也有權取得原始碼。
- 本專案因此**無法在不重寫的前提下改為閉源或商業授權**，這是採用 AGPL 時已知並接受的代價。

### 模型權重

- 上游權重與功能 3 微調產出的權重**都不隨本 repo 散布**，請依 Mortal 專案的規範自行取得。
- 微調權重是否構成 AGPL 意義下的「衍生作品」，目前**法律上並無定論**
  （一派視權重為資料、一派視為訓練程式的產物，尚無判例）。
  本專案採保守立場：若日後散布微調權重，一併以 AGPL-3.0 標示。
  這個不確定性應在論文中如實陳述，而非略過。

### 第三方成果

各第三方專案的授權狀態與本專案的使用方式，逐項記錄於 [NOTICE](NOTICE)。

### 使用條款

**本專案為學術研究用途。** 第三方輔助工具可能違反雀魂使用條款 —— 這與軟體授權是**兩回事**：
授權管的是程式碼怎麼流通，ToS 管的是能不能拿它連遊戲。
展示與實驗建議使用觀戰模式或牌譜回放進行，勿用於排名對局。
