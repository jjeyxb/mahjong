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

### Windows 用 AMD 顯示卡跑（ROCm）

Windows 上 `requirements.txt` 釘的是 PyPI 的 CPU 版 torch，每幀 0.7~2 秒，混合辨識跟不上牌局。AMD 顯示卡加上 `--gpu rocm`，把環境裡的 torch 換成 AMD 官方的 ROCm 版：

```powershell
.venv\Scripts\python.exe tools\setup_yolo.py --gpu rocm
```

- **已經裝好 CPU 版的話直接再跑一次就好。** 換的是 `engines/yolo/.venv` 裡的套件，不搬資料夾，MIA 找環境的路徑不變。跑之前要先關掉 MIA：Windows 上執行中的 torch 檔案換不掉。
- 只支援 Windows + Python 3.12（AMD 只提供這一種 wheel）。Windows 上沒指定 `--python` 時用 `py -3.12` 建環境。
- 第一次要下載數 GB（torch 與 AMD 的 ROCm 執行環境）。版本與雜湊釘在 `engines/yolo/requirements-rocm.txt`；那個檔必須存成 UTF-8 含 BOM，理由寫在檔案開頭。
- 裝完腳本會問一次 torch 看不看得到顯示卡，看不到就報錯，不會安靜地退回 CPU。
- 換回 CPU：`--gpu cpu`，同時移除 ROCm 執行環境。
- 沒加 `--gpu` 時，偵測到 AMD 顯示卡會提示可以加。

RX 9070 實測（同一張牌桌截圖、30 幀）：

| | CPU 版 | ROCm 版 |
|---|---|---|
| 之後每幀 | 約 650 ms 以上 | 約 150~200 ms |
| 第一幀 | 約 2 秒 | 第一次 9.8 秒（現場編譯 GPU 核心），之後約 4 秒 |
| 輸出 | — | 牌河、手牌、寶牌逐項與 CPU 版相同 |

開著 YOLO 時，狀態列會寫 `YOLO · GPU` 或 `YOLO · CPU`。ROCm 版 torch 對外也叫 `cuda`，所以狀態列不照抄 torch 的裝置名稱。macOS 不需要這個選項：Apple Silicon 的 torch 本來就用 MPS。NVIDIA 顯示卡還沒實測過，沒有提供選項。

若權重放在別處：

```sh
.venv/bin/python tools/setup_yolo.py --python python3.12 --models '/path/to/Model'
```

目前這台 Mac 已安裝完成並通過 MPS 推論測試。Windows 用 `.venv\Scripts\python.exe tools\setup_yolo.py`；預設用 `py -3.12` 建環境，要用別的 Python 3.12 就加 `--python <路徑>`。有 AMD 顯示卡的話見上面的 ROCm 一節。

## 使用

1. 依原本方式啟動 MIA（已開啟的舊版本需重新啟動）。
2. 在「向聽分析 → 設定 → 畫面辨識方式」選 YOLO，開啟「畫面辨識」。
3. 網頁版若要 AI 建議走 YOLO，在 AI 建議頁勾選「使用圖像辨識」，再開啟 AI 建議。未勾選時維持封包來源。
4. 放銃分析要比較 YOLO 結果時，勾選原有的「強制使用圖像辨識」。

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
