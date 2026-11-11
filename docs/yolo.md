# YOLO 辨識後端

MIA 可在任一功能頁(AI 建議、向聽分析、放銃分析)的「設定 → 畫面辨識方式」選擇 **樣板比對** 或 **YOLO**。三頁共用同一個值，改一處另外兩頁跟著變。預設為樣板比對，選擇會跨次啟動保存。YOLO 共用既有的遊戲視窗擷取、牌桌框選、手牌顯示、事件 Log、Mortal 與放銃分析介面。

## 安裝

YOLO 使用獨立的 Python 3.12 環境，不把 PyTorch 裝進 MIA 的 Python 3.14 主環境。在專案根目錄執行：

```sh
.venv/bin/python tools/setup_yolo.py --python python3.12
```

腳本建立 `engines/yolo/.venv`，安裝 `engines/yolo/requirements.txt`，並從組員提供的 `downloads/majongsoul ai agent_mac/app/Model` 複製四顆辨識權重與三份旁附 JSON 至 `models/yolo`。模型不進 Git。

權重也公開在 HuggingFace：**[jjeyxb/mia-yolo](https://huggingface.co/jjeyxb/mia-yolo)**。沒有組員那個資料夾的機器，先用 `tools/fetch_models.py` 把權重抓進 `models/yolo`（逐檔核對 SHA-256），再跑上面的腳本 —— 檔案已經在目標位置時腳本不會再去複製：

```sh
.venv/bin/python tools/fetch_models.py
.venv/bin/python tools/setup_yolo.py --python python3.12
```

整個 YOLO 環境要能靠刪掉專案裡的資料夾清乾淨（`engines/yolo/.venv`、`engines/yolo/.cache`、`models/yolo`）：

- `fetch_models.py` 直接寫進 `models/`，不經過使用者目錄下的 HuggingFace 快取。
- 安裝腳本跑 pip 時設環境變數 `PIP_NO_CACHE_DIR=1`，不然 torch 等套件會另外快取一份到使用者目錄（實測約 336 MB）。用環境變數而不是 `--no-cache-dir`：ROCm 的 `rocm` 套件要現場建置，pip 另開的建置子程序看不到命令列參數，照樣會寫快取。
- worker 把 Ultralytics、matplotlib 的設定目錄，以及 AMD 顯示卡第一次跑模型時編譯的 GPU 核心快取，都導到 `engines/yolo/.cache`。

### 用顯示卡跑（GPU）

基本環境裝的是 PyPI 的 CPU 版 torch。Windows 上每幀 0.7~2 秒，混合辨識跟不上牌局。

**最簡單的做法：MIA 的「一般設定 → YOLO 運算裝置」選 GPU。** MIA 會先檢查這台電腦的顯示卡和 YOLO 環境:

- 環境已經能用顯示卡：直接切過去。
- 需要換 torch:顯示要下載哪一版、多大，按「下載並安裝」才開始。
- 顯示卡不支援：說明原因，維持 CPU。

裝不成或取消時都維持 CPU，不會出現「設成 GPU 但其實在跑 CPU」的狀態。

沒選過時的預設：Apple Silicon 是 GPU(MPS 本來就在，以前也一律用它);其他平台是 CPU,因為基本環境裝的是 CPU 版 torch。打包版沒有 `engines/yolo/.venv`,檢查時改成把打包好的推論程序叫起來，看它回報用哪個裝置。

| 顯示卡 | 安裝的 torch | 下載量 | 實測 |
|---|---|---|---|
| Apple Silicon | 不用換（PyPI 版本來就有 MPS） | — | 有 |
| NVIDIA | CUDA 12.8（`requirements-cuda.txt`），驅動要 570 版以上 | 約 2.9 GB | **還沒有** |
| AMD Radeon RX 7700 以上、RX 9000 系列 | Windows ROCm 7.2.1（`requirements-rocm.txt`） | 約 2.0 GB | RX 9070 |
| Intel Arc（獨顯與 Core Ultra 內建） | XPU（`requirements-xpu.txt`） | 約 1.4 GB | **還沒有** |

- 一台電腦有好幾張顯示卡時，依序挑 NVIDIA、AMD、Intel。AMD 不能看到品牌就裝:ROCm 只支援上表那幾代，不在清單上的卡(例如 RX 6600)會直接說明、不下載。
  規則在 `src/mia/yolo/accel.py`,清單來源是 AMD 官方的 Windows 支援表(2026-11-12)。
- 設定頁按下去跑的是 `tools/setup_yolo.py --gpu <cuda|rocm|xpu>`,也可以自己在命令列跑。`--gpu auto` 會照上面的規則自己選。跑之前要先關掉用到 YOLO 的功能，因為 Windows 上執行中的 torch 檔案換不掉;設定頁會擋著，不讓你在 YOLO 執行時安裝。
- 下載期間可以取消，YOLO 環境維持原本那一套。pip 開始換套件之後就不給取消了：裝到一半的 torch 是壞的環境。
- 裝完會由 `engines/yolo/probe.py` 在顯示卡上實際算一次。只看 `is_available()` 不夠：驅動太舊時它照樣回 True,要到第一幀才報錯。算不動就報錯並維持 CPU。
- 換成別的版本時，舊版本帶進來的執行環境(ROCm 的 `rocm-sdk-*`、XPU 的 `intel-*`、`mkl` 等)最後才移除，而且只移除沒有別的套件需要的那些。`--gpu cpu` 會換回 CPU 版。
- 只支援 Windows + Python 3.12(各家只提供這一種 wheel)。Windows 上沒指定 `--python` 時用 `py -3.12` 建環境。版本與雜湊釘在 `engines/yolo/requirements-<版本>.txt`,這幾個檔必須存成 UTF-8 含 BOM,理由寫在 `requirements-rocm.txt` 開頭。
- 選 CPU 時，就算環境裡是 GPU 版 torch 也真的跑 CPU。以前做不到：ultralytics 只要看得到 CUDA 就會用，`MJ_DEVICE=cpu` 擋不住，見 `engines/yolo/README.md` 的本地修改表。
- 打包版不在使用者電腦上跑 pip(`docs/packaging.md`),選 GPU 時會請使用者去「說明 → GPU 加速元件」找發布方提供的元件。

RX 9070 實測(同一張牌桌截圖、30 幀):

| | CPU 版 | ROCm 版 |
|---|---|---|
| 之後每幀 | 約 650 ms 以上 | 約 150~200 ms |
| 第一幀 | 約 2 秒 | 第一次 9.8 秒(現場編譯 GPU 核心),之後約 4 秒 |
| 輸出 | — | 牌河、手牌、寶牌逐項與 CPU 版相同 |

開著 YOLO 時，狀態列會寫 `YOLO · GPU` 或 `YOLO · CPU`。ROCm 版 torch 對外也叫 `cuda`,所以狀態列不照抄 torch 的裝置名稱。

若權重放在別處：

```sh
.venv/bin/python tools/setup_yolo.py --python python3.12 --models '/path/to/Model'
```

目前這台 Mac 已安裝完成並通過 MPS 推論測試。Windows 用 `.venv\Scripts\python.exe tools\setup_yolo.py`；預設用 `py -3.12` 建環境，要用別的 Python 3.12 就加 `--python <路徑>`。要用顯示卡的話見上面的 GPU 一節。

## 使用

1. 依原本方式啟動 MIA（已開啟的舊版本需重新啟動）。
2. 在「向聽分析 → 設定 → 畫面辨識方式」選 YOLO，開啟「畫面辨識」。
3. 網頁版若要 AI 建議走 YOLO，在 AI 建議頁勾選「使用圖像辨識」，再開啟 AI 建議。未勾選時維持封包來源。
4. 放銃分析要比較 YOLO 結果時，勾選原有的「強制使用圖像辨識」。

**雀魂的牌桌與牌背要用預設皮膚。** YOLO 的模型是用預設皮膚的畫面訓練的。2026-11-11 在 Steam 版實測，換成其他牌桌與牌背後，區域模型完全框不到「摸到的牌」，換回預設就正常。這種失敗不會出現任何錯誤訊息，只會看到事件流卡在「與手牌對不上」、AI 建議出不來，所以切到 YOLO 或混合辨識時 MIA 會跳出提醒。提醒框勾「不再顯示」後會記在 `data/ui_state.json` 的 `dismissed_notices`，刪掉那一項就會再跳。

切換辨識方式會停止並重建畫面辨識與使用畫面的功能；正在使用封包的功能不因這個選項重啟。建議在新一局開始前切換，以保留完整開局資訊。YOLO 使用完整牌桌畫面，不使用單獨的手牌框；若校正不準，請框選「整個遊戲畫面」。

畫面辨識、AI 建議與放銃分析共用一個 YOLO 推論程序，不會因同時開啟三個功能而載入三套模型。來源為 YOLO 的 AI 事件 Log 會顯示「圖像辨識 · YOLO」。停止或推論程序失敗時會暫停依賴它的建議。

## 接線

```text
MIA 擷取／牌桌校正 → 最新一幀 → 1920×1080 BGR
  → YOLO 子程序：四模型＋場況／寶牌＋逐幀狀態機
  → 可修訂 Ledger＋穩定手牌快照
  → MIA LedgerStream：檢查輪替、棄牌來源與副露來源、修復起手牌
  → StreamFeeder：前綴有更正時重新同步 Mortal
  → 既有建議／Log／放銃分析
```

MIA 主程序只透過長度標頭與原始影像 bytes 傳遞畫面，取得 JSON 回覆；PyTorch、Ultralytics、裝置設定都留在子程序。擷取端只保留最新一幀，模型忙碌時不堆積歷史畫面。

本次沿用組員的辨識與狀態機來源，保存於 `engines/yolo/agent`。來源檔與雜湊記於 `engines/yolo/upstream.json`。接入點是可更正的事件帳，沒有沿用原本會略過更正、停止整段送出的 MortalGuard。原始副露來源、棄牌 ID 與修訂都會傳給轉接層。

轉接層依回合與來源身分拒絕重複棄牌、重複鳴牌，不以「幾秒內同一張牌」去重，因此不同棄牌來源的兩次相同吃牌仍可成立。對手無法看見的摸牌可以在合法輪替下補為 `?`，自己的未知牌不會送進引擎。每次重新建立整局快照，避免將錯誤永久記入不可逆的送出帳。

自己的起手牌可由同時點的穩定實體手牌，反向扣回已確認摸打與鳴牌重建；重建仍要求 13 張、非負張數、普通牌最多四張、每種赤五最多一張。當局面驗證失敗時，會顯示理由並暫停建議，等待後續畫面更正。

## 驗證：2026-11-06

使用 `human-20261104-185744` 的既有 YOLO 原始觀測，完整重播 7,468 個時間點，再以同場封包評分；封包不參與辨識或轉接。

| 項目 | MIA 整合後最終快照 |
| --- | ---: |
| 起局欄位、自己的配牌 | 35／35、5／5 |
| 捨牌 | 198／198，錯／漏／多均為 0 |
| 自己摸牌 | 49／49（含莊家起手 14 張等價拆分） |
| 副露 | 12／12，無多報 |
| 成立立直、槓後寶牌 | 4／4、1／1 |
| 實際 Mortal 重播報錯 | 0 |

結果位於 `out/yolo-integration-20261106/final-validated/`。另以實際 MPS 子程序對 12 個錄影時間點完成原始影像通訊測試，辨識到手牌，且 MIA 主程序沒有匯入 torch。最終完整測試 2541 passed、9 skipped；ruff 與 mypy 通過。

完整事件結果驗證的是相同預測接上新轉接層後的行為，不是重新推論整場，也未模擬即時取最新幀造成的丟幀。僅驗證一場真人局；部分事件在錄影時間軸上的尾端確認延遲仍較長，不能將最終正確率當成每一手都及時正確。其他皮膚、CPU／MPS 一致性與長時間即時使用仍需擴充素材驗證。

重播既有觀測並實際送至 Mortal：

```sh
.venv/bin/python tools/replay_yolo.py data/recordings/human-20261104-185744 \
  --observations out/yolo-eval-20261105/human-20261104-mps/observations.ndjson \
  --out out/yolo-integration-recheck --engine
```

省略 `--observations` 時會透過真正 YOLO 子程序逐幀重新辨識，耗時較長；省略 `--engine` 可只驗證事件轉接。

## 診斷

模型載入或推論錯誤會顯示在 MIA 狀態列，完整子程序輸出在 `logs/yolo-worker.log`。可以用 `MIA_YOLO_PYTHON` 指定已有的推論 Python，以 `MIA_YOLO_MODELS` 指定模型資料夾。組員版的 `MJ_DEVICE=cpu`／`mps`／`cuda` 裝置選擇亦保留；預設依可用裝置選擇。
