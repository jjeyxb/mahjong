# YOLO 辨識後端

MIA 可在任一功能頁(AI 建議、向聽分析、放銃分析)的「設定 → 畫面辨識方式」選擇 **樣板比對** 或 **YOLO**。三頁共用同一個值，改一處另外兩頁跟著變。預設為樣板比對，選擇會跨次啟動保存。YOLO 共用既有的遊戲視窗擷取、牌桌框選、手牌顯示、事件 Log、Mortal 與放銃分析介面。

## 安裝

YOLO 使用獨立的 Python 3.12 環境，不把 PyTorch 裝進 MIA 的 Python 3.14 主環境。在專案根目錄執行：

```sh
.venv/bin/python tools/setup_yolo.py --python python3.12
```

腳本建立 `engines/yolo/.venv`，安裝 `engines/yolo/requirements.txt`，並從組員提供的 `downloads/majongsoul ai agent_mac/app/Model` 複製四顆辨識權重與三份旁附 JSON 至 `models/yolo`。模型不進 Git。若權重放在別處：

```sh
.venv/bin/python tools/setup_yolo.py --python python3.12 --models '/path/to/Model'
```

目前這台 Mac 已安裝完成並通過 MPS 推論測試。Windows 可用現有啟動 Python 執行同一腳本；`--python` 指向另外安裝的 Python 3.12 執行檔。

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
