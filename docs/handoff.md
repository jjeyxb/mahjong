# 交接文件 — MIA

給接手這個專案的下一個對話 / 下一個人。**最後更新 2026-07-30,commit `4eac468`。**

這份文件只放「接手需要知道」的東西:現況、跑法、已經決定過不要再討論的事、
以及踩過的坑。**決策的完整理由在 [decisions.md](decisions.md)**,不在這裡重抄。

---

## 一分鐘版

**MIA(Mahjong Intelligence Assistant)** 是雀魂麻將輔助軟體,大學畢業專題。
macOS 已實機驗證,Windows 程式寫好但沒機器測過。

三個**互相獨立**的功能:

| # | 功能 | 狀態來源 | 現況 |
|---|---|---|---|
| 1 | 手牌辨識 + 向聽 / 進張 | 螢幕擷取 + CV | ✅ 完成,但只在單一解析度驗過 |
| 2 | AI 建議(Mortal) | WebSocket 封包 → MJAI | ✅ 完成,已實機打過三場 |
| 3 | 打法風格微調權重 | 離線訓練 | ❌ 未動工,被 GRP 權重擋住 |

解耦是刻意的:功能 1 只吃畫面、功能 2 只吃封包,任何一個做不完不會擋住其他。

**現在可以真的拿來用**(`--live`)。805 個測試、ruff + mypy 乾淨。

---

## 馬上能跑

```bash
cd /Users/caoyunjie/project/mahjong

# 實際使用:瀏覽器會自己開,登入後撥開視窗上的開關
.venv/bin/python tools/ui.py --live --mortal models/mortal_298k.pth \
    --user-data-dir data/live/chrome-profile

# 不開遊戲就能看 UI(重播真實對局,引擎真的在跑)
.venv/bin/python tools/ui.py --replay tests/fixtures/real_game_full.jsonl \
    --mortal models/mortal_298k.pth --speed 3

# 只看版面
.venv/bin/python tools/ui.py --demo

# 全部檢查
.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q
```

操作細節與**狀態列訊息對照表**在 [live.md](live.md) —— 即時模式出問題時先看那張表。

### 環境

| | |
|---|---|
| 主程式 | `.venv` — Python **3.14.6** |
| AI 引擎 | `engines/mortal/.venv` — Python **3.12.13**(獨立) |
| 權重 | `models/mortal_298k.pth`(130MB,gitignore),tag `mortal4-b40c192-t26031702` |
| 上游 | `engines/mortal/Mortal/`(clone,gitignore) |

**為什麼兩個環境:** `libriichi` 不在 PyPI、`mjai` 只有 cp312 wheel,兩者在
3.14 都裝不起來。這是子程序架構的**技術**原因 —— 不是授權隔離(見下)。

---

## 架構

### 三個進程

```
┌──────────────────────────────────────────┐
│ 主程式 (Python 3.14 · .venv)              │
│ capture → vision → analysis → ui          │
│ groundtruth → mjai → engine → ui          │
└───────┬──────────────────────┬───────────┘
        │ JSON-lines / stdio   │ 錄影檔 (jsonl)
        ▼                      ▼
┌───────────────────┐   ┌──────────────────────┐
│ AI Engine ×N      │   │ tools/gt.py cdp      │
│ (Python 3.12)     │   │ Playwright + CDP     │
│ Mortal + libriichi│   │ 寫 ws.jsonl          │
└───────────────────┘   └──────────────────────┘
```

### 即時模式的三條執行緒

```
擷取執行緒(15 fps)──┐
                      ├─▶ UpdateBus ──▶ pump()(UI 執行緒)──▶ ViewModel ──▶ 視窗
封包執行緒(事件驅動)─┘
```

`pump()` 是**唯一**碰 ViewModel 的地方,只在 UI 執行緒跑。這件事必須成立:
Qt widget 只能在建立它的執行緒上動,而 ViewModel 自己不是執行緒安全的。
所以 `LiveRuntime` 一個鎖都不需要。

`mia.live` 與 `mia.ui.viewmodel` **都不 import Qt** —— 整條即時路徑沒有事件迴圈
也測得動。

---

## 已經決定過的事(不要重新討論)

每一條的完整理由在 decisions.md,這裡只留結論與一句話的why。

| 決定 | 一句話 |
|---|---|
| **封包當線上狀態來源,不只當 GT** | 專案中期改的。原本 CV 要重建整個牌桌,撞到三道牆(見 decisions.md 第二節) |
| **CV 只認自己的手牌** | 副露內容不必辨識 —— 向聽只要知道組數,而組數可由張數反推 |
| **AGPL-3.0** | 主動選的,不是被 Mortal 傳染的。所以「子程序 = 獨立作品」這個論證不必成立 |
| **子程序不是為了授權隔離** | 是因為 libriichi 裝不進 3.14。因果方向不要寫反(README 有專章) |
| **郵箱而不是佇列** | UI 卡住要掉幀,不要恢復後補播已經不成立的局面 |
| **手牌與建議是兩個 slot** | 合併的話別人打牌的事件會蓋掉還該顯示的建議 |
| **擷取子程序不綁在開關上** | 綁上去關掉再打開會殺掉瀏覽器,整場對局就沒了 |
| **兩個功能預設關閉** | 開它們會載 130MB 權重 / 開始持續擷取螢幕,該是明確動作 |
| **peek 用「砍掉重練」復原,不做記帳式最佳化** | 記帳要猜使用者下一步,猜錯是整場拿錯局面**而且不報錯** |
| **從錄影檔頭讀而不是從尾巴接** | 座位、寶牌、誰立直了都只在先前的事件裡 |
| **PySide6 而非 PyQt6** | 原因(LGPL)已隨 AGPL 決定失效,但沒有換的理由 |

---

## 實測數字(散在各處,這裡集中)

### 引擎

| 項目 | 數字 |
|---|---|
| Mortal 啟動到握手 | **0.50 ~ 0.64 秒**(連續三次) |
| 每個決策的推論 | 10 ~ 23 ms |
| baseline(規則式) | < 1 ms |
| `peek()` 一次(重啟 + 重播) | 1.30 / 1.54 秒 |
| 一場東風戰會 peek 幾次 | 2 次 |
| 全場 550 事件重播 | 2.0 秒(含 peek 3.8 秒) |

> ⚠ 舊文件曾寫「載權重約 10 秒」並以此當設計理由 —— **那是錯的**,從沒實際
> 計時過。已全部改正。這是「先量再說」的反例。

### 與真人的一致率(`tools/advise.py`)

| 素材 | 決策點 | `mortal_298k` | baseline | 兩者彼此 |
|---|---|---|---|---|
| 完整東風戰(550 事件) | 73 | **66%** | 42% | 53% |
| 早期節錄(249 事件) | 33 | **76%** | 39% | 42% |

加上 8 個「跳過」決策後,總決策點是 **81**(73 個有動作 + 8 個跳過)。

**一致率不是準確率** —— 人打錯時引擎不跟著錯,這個數字反而下降。

### libriichi 動作空間(ACTION_SPACE = 46,實測驗證)

```
0–8 = 1m–9m   9–17 = 1p–9p   18–26 = 1s–9s
27–33 = 東南西北白發中        34–36 = 赤五 5mr/5pr/5sr
37 = 立直   38–40 = 吃(三種)   41 = 碰   42 = 槓
43 = 和了   44 = 流局   45 = 跳過
```

42 與 43 沒直接探到,但前後夾死。**38/39/40 三種吃無法從遮罩分辨是哪一種吃法**
—— 要看動作本身的 `consumed`。

### CV

| 項目 | 數字 |
|---|---|
| 牌面模板 | 37 張 PNG,64×109,`assets/tiles/mjpface_default/` |
| 佔用判定(灰階標準差) | 正常牌 55.3–94.9、白板 35.6、空桌布 21.1–23.0 |
| 單幀校正的不穩定度 | 216 幀產生 **21 種**不同 table_rect,約 16% 明顯錯誤 |
| `own_hand` ROI | `[0.1150, 0.8533, 0.7083, 0.1437]`,量自 2560×1440 |

### 郵箱 / 手牌追蹤

| 項目 | 數字 |
|---|---|
| 一場對局:投遞 → UI 通知 | 210 筆 → **12 次**(合併掉 188) |
| `HandTracker` 完整一場 | 137 個手牌狀態、**0 次脫節**、0 個不合法副露組數 |
| 錄影解析 | 1398 frame → 550 MJAI 事件,**100%** 成功 |
| DumpTail 半截行處理 | 1398 行、**0 個壞行**(刻意在第 400 行切一次) |

---

## 素材

| 路徑 | 內容 |
|---|---|
| `tests/fixtures/real_game_full.jsonl` | **主要 fixture。** 完整東風戰,1398 frame / 550 事件 / 5 局 / 有 `end_game` 與加槓 |
| `tests/fixtures/hand_*.png` | 三張真實手牌截圖(13 張滿手 / 14 張含摸牌 / 4 張含副露),皆 1813×207 |
| `data/live/20260730-193801/ws.jsonl` | 2026-07-30 實機第二場,234 事件、2 局、**含立直**、100% 解析 |
| `data/live/20260730-192227/ws.jsonl` | 同日第一場,65 事件 |

`data/` 與 `models/` 都在 gitignore。

---

## 踩過的坑(別再踩一次)

這些都花了真實時間,而且**大多是渲染 / 實機才看得到,單元測試不會抱怨**。

### UI / Qt

* **window flag 必須在 `show()` 之前設。** show 之後改,Qt 會把視窗隱藏並要求
  重新 show —— 症狀是「Dock 有圖示但畫面上沒有視窗」,完全不像 flag 造成的。
* **`QPropertyAnimation` 要靠事件迴圈推進。** 建構期迴圈還沒起來,動畫一格都不會
  走。開關會停在錯的一邊(看起來是關的,但功能在跑)。用 `set_initial()`。
* **截圖腳本一定要 `processEvents()`。** 版面配置與動畫都要靠它。不推進的話會
  看到「手牌整段消失」這種假象 —— 我為此誤判過一次,以為是自己剛改壞的。
* **`isVisible()` 對沒 show 過的視窗的子元件一律回 False。** 測試要用 `isHidden()`。
* **QPixmap 沒有 QApplication 會 segfault**,不是拋例外。
* **`lru_cache` 不要掛在方法上** —— 它會持有 `self`。

### 引擎 / 協定

* **`StartGame.names` 不能是 `[]`** —— libriichi 要求「可以沒有,但有就必須四個」。
* **`{"type":"none"}` 有兩種意思**,靠 `mask_bits` 位元數才分得開(見第十二節)。
  這個 bug 讓「碰 / 槓 / 跳過」時畫面一片空白。
* **立直之後切哪張要另外問** —— 引擎回 `reach` 時不會說(見第十三節)。
* **吃牌一定要顯示 `consumed`** —— 不只是顯示問題,`action_label` 也用來判斷
  引擎分歧,不寫的話不同吃法會被當成一致。
* **綁定方法每次取用都是新物件** —— `sink is self._queue.put` 永遠 False。
  這個 bug 讓「子程序死掉」被誤報成「逾時」。
* **鳴牌之後沒有摸牌**(`drawn is None`)但手上是 11 張、還欠一張沒打。
  用 `drawn` 判斷副露組數會每次鳴牌後算錯 —— 要用 `len(tiles) % 3`。

### 其他

* **手牌排序不能用 `sorted()`** —— 會把筒子插進萬子中間。用 `tiles.sort_key`。
* **視窗自動偵測會選到編輯器** —— 開著這個專案的編輯器標題含「雀魂」而且視窗
  更大。錄製時一定要 `--window` 指定 handle。
* **`prelude` 會 import tensorboard**,而且**不會**動 `sys.path`。bot.py 自己插。
* **stdout 重導到檔案時是 block-buffered** —— `print` 會等到程式結束才落地。
  除錯時別因此以為程式沒跑。

---

## 未解 / 沒驗過的

| # | 項目 | 影響 | 備註 |
|---|---|---|---|
| 1 | **畫面 + 封包同時在真實對局跑** | 兩條路各自驗過,沒一起跑過 | 需實機打一場 |
| 2 | **`own_hand` ROI 在其他解析度** | 座標量自 2560×1440;實機是 1019×804 | 正規化理論上能縮放,沒驗 |
| 3 | **M3-1b:per-tile 準確率報告** | 工具鏈寫好了,缺成對素材 | **使用者明確暫緩** |
| 4 | `StableCalibrator` 端對端複驗 | 已修但沒對真實錄影驗 | 可與 #3 一起解決 |
| 5 | MITM 路線 CA 憑證 | 雀魂是 Unity,是否信任系統憑證未知 | **使用者要求延後** |
| 6 | Windows 擷取層 | 程式寫好,沒機器測 | |
| 7 | 沒有公開 GRP 權重 | 擋住功能 3 | 須先跑 `train_grp.py` |
| 8 | 受入枚數略微高估 | 不知副露內容導致 | 次要,可從封包補 |
| 9 | 單一 session 內視窗縮放 | manifest 只存一個 `table_rect` | 要改成存進每個 `FrameRecord` |
| 10 | UI 皮膚切換 | 要重載跨 widget 的 `TileIcons` | 未實作 |

**沒開始也沒被授權開始的:** M7 Overlay、功能 3 離線訓練。

---

## 使用者的偏好與指示

* **MITM CA 憑證測試已明確延後**,不要主動去做。
* **M3-1b 錄製已明確暫緩**(「錄製先放一邊」),要花約 20 分鐘實際遊玩。
* 大的或有歧義的範圍變動**先問**;但同一條工作線裡連續的「開始!」「動手吧」
  就是繼續做的意思。
* UI 版面參考明日方舟的 MAA:**左側直排功能列 + 每頁頂端放該頁自己的設定**。
* 開關要 **iOS 樣式的綠色**(用 Apple 的 system green `#34C759`)。
* 文件與註解用**繁體中文**,語氣直接、講清楚 why 而不只是 what。

---

## 檔案地圖

```
src/mia/
├── features.py     兩個功能的識別字(ui 與 live 共用,無相依)
├── config/         pydantic 設定 + YAML 載入
├── capture/        macos(Quartz) / windows(PrintWindow) / mss fallback
├── calibration/    牌桌矩形偵測(stable.py 是多幀中位數穩定化)
├── vision/         純 CV 無狀態:roi + tiles/(hand 定位、classify 分類)
├── analysis/       向聽 / 進張 / 打牌建議(mahjong 套件)
├── mjai/           events / tiles / handstate(HandTracker)
├── engine/         base / subprocess_engine / mortal / dummy / multiplex / actions
├── live/           bus / vision / packets / runtime / source  ← 即時模式
├── ui/             viewmodel + panel/ widgets/(tiles, advice, analysis, toggle)
├── groundtruth/    cdp / capture_addon / dump / liqi / schema / to_mjai / stream
├── eval/           align / report(準確率評測,素材待補)
└── recorder/       錄製資料集、離線回放

tools/  ui.py(側邊視窗)  gt.py(封包擷取/檢視)  advise.py(離線重播給引擎)
        evaluate.py(準確率)  record.py(畫面錄製)  roi_annotate.py  fetch_tiles.py

docs/   decisions.md  ← 方向轉折與全部決策理由(最重要)
        live.md       ← 即時模式操作手冊 + 狀態列對照表
        recording.md  ← 錄製資料集的檢查清單
        handoff.md    ← 這一份
```

### 讀原始碼的建議順序

1. `README.md` 的「架構」與「兩條資料流」
2. `docs/decisions.md` 第二、三節(為什麼是這三個功能)
3. `src/mia/ui/viewmodel.py` —— 整個 UI 要顯示什麼都在這裡,而且不碰 Qt
4. `src/mia/live/bus.py` 的模組 docstring —— 郵箱的設計理由
5. `src/mia/engine/base.py` —— 引擎介面與 `Advice` 的語意

每個模組的 docstring 都寫了「為什麼這樣做」與「刻意不做什麼」,那是主要的
設計文件 —— 讀它們比讀程式碼快。
