# 交接文件 — MIA

給接手這個專案的下一個對話 / 下一個人。**最後更新 2026-08-26(放銃分析 + 準備移到 Windows)。**

> 🪟 **要在 Windows 上接手的話,先看 [在 Windows 上重建](#在-windows-上重建)。**
> 那一節列出 git 拉不到、必須手動補的四樣東西,以及第一件該做的事。

這份文件只放「接手需要知道」的東西:現況、跑法、已經決定過不要再討論的事、
以及踩過的坑。**決策的完整理由在 [decisions.md](decisions.md)**,不在這裡重抄。

---

## 一分鐘版

**MIA(Mahjong Intelligence Assistant)** 是雀魂麻將輔助軟體,大學畢業專題。
macOS 已實機驗證,Windows 程式寫好但沒機器測過。

### ⚠ 「功能 3」有兩個意思,不要搞混

文件裡有兩套編號在跑,這是歷史累積下來的:

| 說法 | 指的是 | 在哪 |
|---|---|---|
| **App 的三個開關** | 畫面辨識 / AI 建議 / **放銃分析** | `features.py`、UI 上看得到的那三個 |
| **專題的 M8「功能 3」** | **打法風格微調**(離線訓練) | README 的里程碑表 |

`features.py` 的 `DANGER` 註解寫「功能 3」,指的是第一種。README M8 的「功能 3」
是第二種。兩者**完全不同**,而且都還在用 —— 講的時候用名字,不要用編號。

### App 的三個開關

| # | 開關 | 狀態來源 | 現況 |
|---|---|---|---|
| 1 | 手牌辨識 + 向聽 / 進張 | 螢幕擷取 + CV | ✅ 完成,但只在單一解析度驗過 |
| 2 | AI 建議(Mortal) | WebSocket 封包 → MJAI | ✅ 完成,已實機打過四場 |
| 3 | **放銃分析**(每張牌切出去的危險度) | WebSocket 封包 → 桌面狀態 | ✅ 完成,已實機驗過一場 |

解耦是刻意的:1 只吃畫面,2 與 3 吃封包但**只有 2 會載 130MB 權重** ——
所以想看放銃分析的人不必付那個代價。任何一個做不完不會擋住其他。

### 專題里程碑

M0–M7 全部 ✅。**只剩 M8:打法風格微調**,被「沒有公開的 GRP 權重」擋住,
必須自己先跑 `train_grp.py`。那是整個專案剩下唯一「可能根本做不成」的部分,
而且要 GPU —— 所以才要搬到 Windows(RX 9070 + ROCm)。

**兩種呈現方式:** 側邊視窗(M6)與 Overlay(M7,疊在遊戲上的精簡 HUD)。
兩者訂閱同一個 `ViewModel`,顯示邏輯不重複實作。

**現在可以真的拿來用**(`--live`)。**1121 個測試**、ruff + mypy 乾淨。
最後一個 commit `1160089`,已推上 `origin/main`。

---

## 馬上能跑

macOS:

```bash
cd /Users/caoyunjie/project/mahjong

# 實際使用:按左下角「開始遊戲」開瀏覽器,登入後撥開視窗上的開關
# 登入狀態預設留在 data/live/chrome-profile,不必再加旗標
.venv/bin/python tools/ui.py --live --mortal models/mortal_298k.pth

# 不開遊戲就能看 UI(重播真實對局,引擎真的在跑)
.venv/bin/python tools/ui.py --replay tests/fixtures/real_game_full.jsonl \
    --mortal models/mortal_298k.pth --speed 3

# 只看版面
.venv/bin/python tools/ui.py --demo

# 全部檢查
.venv/bin/python -m ruff check . && .venv/bin/python -m mypy && .venv/bin/python -m pytest -q
```

Windows(PowerShell)—— 只有直譯器路徑不同,參數完全一樣:

```powershell
cd C:\path\to\mahjong

.venv\Scripts\python.exe tools\ui.py --live --mortal models\mortal_298k.pth

# 不需要遊戲、不需要權重就能驗 UI 有沒有跑起來(建議 Windows 上第一個先跑這個)
.venv\Scripts\python.exe tools\ui.py --demo

.venv\Scripts\python.exe -m ruff check . ; .venv\Scripts\python.exe -m mypy ; .venv\Scripts\python.exe -m pytest -q
```

操作細節與**狀態列訊息對照表**在 [live.md](live.md) —— 即時模式出問題時先看那張表。

---

## 在 Windows 上重建

git clone / pull 之後**還缺四樣東西** —— 它們都在 gitignore 裡,拉不到:

| 缺什麼 | 怎麼補 | 大小 |
|---|---|---|
| `models/mortal_298k.pth` | `curl -L -o models/mortal_298k.pth https://huggingface.co/VoidShine/mortal-298k/resolve/main/mortal_298k.pth` | 130 MB |
| `engines/mortal/Mortal/` | `git clone --depth 1 https://github.com/Equim-chan/Mortal.git engines/mortal/Mortal` | |
| Playwright 的 Chromium | `.venv\Scripts\python.exe -m playwright install chromium` | |
| 兩個 venv | 見下 | |

`data/` 空的沒關係 —— 錄影檔會自己長出來,而測試要用的素材在
`tests/fixtures/`(有進 git)。

### 兩個環境

```powershell
# 主程式 — Python 3.14
py -3.14 -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\pip install -e . --no-deps   # tests/conftest.py 直接 import mia,沒這行 pytest 收集階段就 ModuleNotFoundError

# AI 引擎 — Python 3.12,獨立
py -3.12 -m venv engines\mortal\.venv
engines\mortal\.venv\Scripts\pip install -r engines\mortal\requirements.txt
```

> `pip install -e .` 這一步在 macOS 上也是必須的,只是先前的文件漏寫了 ——
> 這次在 Windows 重建時才發現 `mia` 從沒被真的裝進 venv 過(`requirements-dev.txt`
> 裝的都是相依,不含專案本身)。`requirements-dev.txt` 已經把這一步寫進開頭的
> 註解,上面這行是給不會去讀 txt 檔頭的人看的。mypy 也少了兩個 stub
> (`types-PyYAML`、`types-protobuf`),已經一併補進 `requirements-dev.txt`。

**為什麼兩個環境:** `libriichi` 不在 PyPI、`mjai` 只有 cp312 wheel,兩者在
3.14 都裝不起來。這是**技術**原因,不是授權隔離(見下)。

### libriichi 的 Windows 產物名稱與 macOS 不同

```powershell
cd engines\mortal\Mortal
cargo build -p libriichi --lib --release
copy target\release\riichi.dll mortal\libriichi.pyd
```

三件事會咬人:

* **產物叫 `riichi.dll`,不是 `libriichi.dll`** —— libriichi 的 `[lib] name` 是 `riichi`。
* **副檔名要改成 `.pyd`** —— CPython 在 Windows 上只認 `.pyd`。
  (macOS 是 `.dylib` → `.so`,同一個坑的不同平台版本。)
* ~~Rust 建置需要 MSYS2~~ —— **已在真機上驗證過是錯的。** rustup 預設的
  `x86_64-pc-windows-msvc` 工具鏈直接就能建,完全沒裝 MSYS2 的 mingw-w64
  子系統。要用的是 **MSVC Build Tools**(`vcvars64.bat` 把 `link.exe` 放進
  PATH),不是 MSYS2。這條原本是沒上機測過的猜測,已更正。

```powershell
# 沒開過 "Developer PowerShell for VS" 的話,先跑這行把 MSVC 工具鏈掛進 PATH
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && powershell'
```

`PYO3_PYTHON` 要指到 3.12 那一個直譯器,不然會編出對不上的 ABI。實測
(cargo 1.95、rustc 1.95、release profile)**66 秒**建完,import 與跑真的推論
都正常,一致率數字與 macOS 記錄的完全一樣(見下面「實測數字」)——這代表
Windows 建出來的 libriichi 行為上與 macOS 那份等價,不是巧合過關。

### 第一件該做的事:先撞最貴的失敗

順序刻意**不是**「先讓 App 跑起來」:

1. **M8 冒煙測試** —— 建 libriichi + 載入 298k 權重 + 跑一次推論。
   README 自己寫了「這一步同時驗證建置、GPU 環境、權重相容性三件事,
   失敗的話也是最早、最便宜的失敗點」。
2. **ROCm 認不認得 RX 9070**、PyTorch 走不走得到它。
3. 上面兩關過了,才回頭驗擷取層。

理由:擷取層失敗是「要修程式」,可以修;ROCm / libriichi 失敗是「要換方向」,
那個越早知道越好。

### 擷取層:三個已知風險

`src/mia/capture/windows.py` 的模組 docstring 自己列了(它從沒在真機上跑過):

```powershell
python tools\capture_probe.py --list --capture ...
```

1. **`PW_RENDERFULLCONTENT` 對硬體加速視窗可能抓到全黑。** Chrome / Edge /
   Electron 都是。全黑的話改用 `dxcam`(已在 requirements)或 Windows
   Graphics Capture。
2. **DPI 縮放下 `DwmGetWindowAttribute` 的邊界可能與影像尺寸對不上。**
   程式啟動時會設 `PER_MONITOR_AWARE_V2`,但沒實測過。
3. **視窗被遮擋時抓不抓得到完整內容。** macOS 上實測是可以的(連完全遮住都行),
   Windows 未知。

⚠ 若最後退回 `dxcam`,它抓的是**螢幕區域**而不是視窗緩衝區 —— 那條路上
**Overlay 會被拍進辨識輸入**,要讓 HUD 避開手牌區。`capture/factory.py`
退回 mss 時會警告,dxcam 這條要自己記得。

### 跑得起來但要留意的

* **視窗自動偵測會選到編輯器** —— 開著這個專案的 VS Code 標題含「雀魂」,
  而且視窗更大。錄製時一律 `--window` 指定 handle。這在 macOS 上抓到過 Safari。
* 測試裡有 `platform_windows` marker,但**目前沒有任何一條真的針對 Windows**
  —— 別把「測試全過」當成擷取層可用。

### 版本對照(macOS 上實際在跑的,Windows 照這個裝)

| | |
|---|---|
| 主程式 | `.venv` — Python **3.14.6** |
| AI 引擎 | `engines/mortal/.venv` — Python **3.12.13**(獨立) |
| 權重 | `models/mortal_298k.pth`(130MB,gitignore),tag `mortal4-b40c192-t26031702` |
| 上游 | `engines/mortal/Mortal/`(clone,gitignore) |

---

## 架構

### 三個進程

```
┌──────────────────────────────────────────┐
│ 主程式 (Python 3.14 · .venv)              │
│ capture → vision → analysis → ui          │
│ groundtruth → mjai → engine → ui          │
│ 「開始遊戲」按鈕開下面那個子程序          │
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
擷取執行緒(15 fps)──┐                                              ┌─▶ 側邊視窗
                      ├─▶ UpdateBus ──▶ pump()(UI 執行緒)──▶ ViewModel
封包執行緒(事件驅動)─┘                                              └─▶ Overlay
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
| **擷取子程序不綁在功能開關上** | 綁上去關掉再打開會殺掉瀏覽器,整場對局就沒了。它綁在「開始遊戲」按鈕上 |
| **瀏覽器等按鈕,不自己開** | 與兩個功能預設關閉同一個原則;而且原本瀏覽器一關就沒辦法再開一場,只能重開 MIA |
| **每按一次「開始遊戲」換一個錄影檔** | 錄影檔是 append 開的、封包是從檔頭讀的 —— 兩場疊在一起會拿著結束的牌局給建議**而且不報錯** |
| **按鈕沒有「結束遊戲」** | 關瀏覽器等於丟掉那一場,不該是手滑按得到的東西 |
| **兩個功能預設關閉** | 開它們會載 130MB 權重 / 開始持續擷取螢幕,該是明確動作 |
| **peek 用「砍掉重練」復原,不做記帳式最佳化** | 記帳要猜使用者下一步,猜錯是整場拿錯局面**而且不報錯** |
| **從錄影檔頭讀而不是從尾巴接** | 座位、寶牌、誰立直了都只在先前的事件裡 |
| **PySide6 而非 PyQt6** | 原因(LGPL)已隨 AGPL 決定失效,但沒有換的理由 |
| **Overlay 手動拖曳,不自動跟隨遊戲視窗** | 跟隨要一條輪詢 timer,而且 macOS 取視窗標題要螢幕錄製權限 —— 只用 AI 建議的人不該為了 HUD 定位被逼著給 |
| **Overlay 的三個控制項都在側邊視窗** | 它鎖定後對滑鼠透明,擺在自己身上的按鈕會變成看得到卻按不到 |
| **Overlay 不進 features.py 的開關體系** | 那兩個開關管的是 130MB 權重與持續擷取螢幕;Overlay 只是換一種畫法,沒有那個成本 |
| **視窗位置存 `data/ui_state.json` 而非 config/** | `save_config()` 寫的是整份快照,會把當下所有預設值一起寫死,日後改預設不生效**而且沒有症狀** |
| **放銃分析輸出「還剩幾種待牌型」,不編機率** | 「放銃率 12.3%」要有語料才有意義,手上兩場差三四個數量級。編一個數字出來會得到「看起來精確、實際沒來源」的東西 |
| **放銃分析評估三家,不是只算立直的** | 實測那場三次榮和有**兩次**是沒立直的人和的。只算立直家會對真正的放銃牌說「安全 —— 現物」 |
| **三副露算威脅,但不動危險度等級** | 平起平坐的是**指名**。等級由排除法數出來、副露是估計,混在一起畫面會出現「非常危險」配「還有 2 型」 |
| **放銃分析不載引擎** | 它要的是事件流本身。與 AI 建議分成兩個開關,就是為了讓人能只開這個而不付 130MB 的代價 |
| **每一列都要寫「對誰」** | 一張牌對立直那家是現物、對另一家全新。只寫「安全」使用者就會照著打 —— 真實牌譜裡發生過 |
| **Overlay 的放銃清單與功能開關分開勾** | 功能開著代表有在算,勾選才決定要不要占掉 HUD 的高度(一整手 14 列) |

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

### 放銃分析(全部量自 `real_game_full.jsonl`,259 次捨牌)

| 項目 | 數字 |
|---|---|
| 三次榮和的和牌者 | 3 副露(沒立直)/ 2 副露(沒立直)/ 立直 —— **兩次沒立直** |
| 副露數分布(捨牌時點 × 四家) | 0 組 786、1 組 47、2 組 128、**3 組 75** |
| 曝光量:三副露 vs 立直 | **53 vs 59** 個捨牌時點 —— 同一個量級 |
| 有人被指名的比例:只認立直 | 47/259(**18%**) |
| 同上:門檻 3 副露(現行) | 100/259(**39%**) |
| 同上:門檻 2 副露 | 132/259(51%)—— 標籤開始貶值,故未採用 |
| 每張牌的危險度分布 | 安全 4.2% / 比較安全 40.9% / 危險 35.1% / 非常危險 19.7% |

**我們自己那次放銃(8s)**,和牌者只有 2 副露 —— 現行門檻 3 抓不到他。
那是已知的覆蓋缺口,見「未解」#13。

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
| `data/live/20260817-100915/ws.jsonl` | 2026-08-17 實機第四場,319 frame / 186 秒。**放銃分析第一次實機**,東 1 局 |

⚠ `data/` 在 gitignore —— 上面三份**不會**跟著 git 到 Windows。要的話手動搬,
不搬也沒關係:`tests/fixtures/real_game_full.jsonl` 才是被測試依賴的那一份。

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
* **`WA_TransparentForMouseEvents` 會偷偷改 window flag。** 設 True 時 Qt 自己把
  `WindowTransparentForInput` 加進 flag,設回 False **不會**收回去 —— 症狀是
  Overlay 鎖過一次就再也解不開,而「拖不動」完全不像是 flag 的事。只動
  window flag 就好,attribute 是多餘的。
* **`Qt.Tool` 在 macOS 上會在程式失去焦點時自動隱藏** —— 而 Overlay 的正常
  使用情境正是「焦點在遊戲上」。要 `WA_MacAlwaysShowToolWindow`。
* **offscreen 平台不支援 `propagateSizeHints()`** —— 用 `QLayout.SetFixedSize`
  自動縮放的視窗,在無頭截圖裡尺寸可能不會跟著內容變。那是平台的事,不是
  版面壞了。

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

### 只有實機 / 渲染才看得到的(這一類最貴)

* **「未開啟」那句話的條件寫死了兩個功能。** 加了放銃分析之後,它開著、
  側邊視窗滿滿是資料,Overlay 照樣喊「未開啟 —— 用側邊視窗的開關打開」。
  最糟的是**它把人指向錯的開關**:去撥一個已經開著的東西,撥完沒反應。
  側邊視窗狀態列同一個成因。改成逐一問 `features.NAMES`。
* **待牌型的名字混進日文新字體** —— `両面` / `単騎` / `辺張`。單元測試不會
  抱怨,要看得懂中文的人盯著畫面才會發現。已改成 兩面 / 單騎 / 邊張。
* **同一個牌名在不同模組是不同記法。** 放銃分析那條路的資料來自 MJAI 事件流
  (`E`、`5mr`),向聽分析那條來自 classify(`1z`、`0m`)。多轉一次
  `ms_to_mjai` 的話**數牌剛好轉得過去、字牌會拋 `TileError`** ——
  所以只有摸到字牌那一刻才會炸。離屏渲染才抓到的。
* **視窗邊界會過期。** `WindowInfo` 是找視窗那一刻抓的,而瀏覽器之後被改過
  大小 —— `scale` 於是算錯,校正永遠鎖不上。三個擷取後端都要每幀重查一次。
* **滑鼠移到待選牌上,牌會往上抬約 32%。** 抬起來的牌超出 ROI,整手就認錯了。
  解法是切 ROI 時多留 `HOVER_HEADROOM = 0.5` 的上緣。
* **深色模式下 `palette(mid)` 比背景還暗。** 那是 3D 陰影的角色,不是文字的。
  用 `palette(placeholder-text)`(由文字色推出來,兩種模式都成立)。

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
| 10 | UI 皮膚切換 | 要重載跨 widget 的 `TileIcons` | 未實作。Overlay 又多一個 `TileIcons` 持有者 |
| 11 | ~~Overlay 實機疊在對局上~~ | ✅ 2026-08-17 驗過,疊在牌桌上正常 | **但放銃清單那一段還沒實機看過** —— 那次沒勾設定頁的選項 |
| 12 | Overlay 自動跟隨遊戲視窗 | 現在是手動拖 | 刻意留到之後,理由見 decisions.md 第十四節 |
| 13 | **放銃分析不看打點** | 放銃給門清立直 dora 3 與給 1000 點的仕掛け同一級 | 量的只有「會不會中」,沒有「中了多痛」 |
| 14 | **副露只用來數壁,沒當危險訊號**(門檻 3 除外) | 染手、役牌碰完全沒被讀 | 染め手 是最可靠的讀之一,而且是**推論不是估計**,值得做 |
| 15 | 三副露門檻漏掉 2 副露 | 我們自己那次放銃的和牌者只有 2 副露 | 「2 副露 + 役牌」比單純的副露數硬,留作獨立一項 |
| 16 | `Player.reach_turn` 記了沒人用 | 無 | 立直後的安全牌是在捨牌當下就記進 `safe`,不需要它 |
| 17 | **`decisions.md` 第十八節沒寫** | 放銃分析的設計理由目前只在程式的 docstring 裡 | 該記的:只認立直會漏掉、等級 vs 指名的界線、三副露的數據 |

**沒開始也沒被授權開始的:** M8 風格微調的離線訓練。

### 未解 #8「受入枚數略微高估」現在變便宜了

`shanten.py:221` 算的是 `COPIES - array[index]` —— **只扣自己手上那幾張**,
牌河、副露、寶牌指示牌全當成還在山裡。而做放銃分析時蓋的
`TableTracker.remaining()` 算的正好是缺的那一半。

所以現在不是「要寫一個追蹤器」,是「把已經有的接過去」。要決定的一點:
功能 1 靠 CV、功能 2/3 靠封包,兩條路目前解耦 —— 接上去等於讓向聽分析在
封包沒開時退回舊行為。建議做成有封包就用、沒有就照舊,畫面上分開
「進張 12」與「進張 12(估)」。

---

## 使用者的偏好與指示

* **MITM CA 憑證測試已明確延後**,不要主動去做。
* **M3-1b 錄製已明確暫緩**(「錄製先放一邊」),要花約 20 分鐘實際遊玩。
* 大的或有歧義的範圍變動**先問**;但同一條工作線裡連續的「開始!」「動手吧」
  就是繼續做的意思。
* UI 版面參考明日方舟的 MAA:**左側直排功能列 + 每頁頂端放該頁自己的設定**。
* 開關要 **iOS 樣式的綠色**(用 Apple 的 system green `#34C759`)。
* 文件與註解用**繁體中文**,語氣直接、講清楚 why 而不只是 what。
  **注意日文新字體**:麻將術語很容易寫成 `両面` / `単騎` / `辺張`,
  要用 兩面 / 單騎 / 邊張。(嵌張、雙碰兩種寫法一致。)
* 切牌建議講**牌名**不講代號 —— 說「東」「南」,不說 `1z`、`2z`。
  `mia.mjai.tiles.tile_name()` 兩種記法都收。
* 下一步已經談定:**搬到 Windows**,先撞 M8 的冒煙測試(libriichi + ROCm),
  再回頭驗擷取層。理由見上面「在 Windows 上重建」。

---

## 檔案地圖

```
src/mia/
├── features.py     三個開關的識別字(ui 與 live 共用,無相依)
├── config/         pydantic 設定 + YAML 載入
├── capture/        macos(Quartz) / windows(PrintWindow) / mss fallback
├── calibration/    牌桌矩形偵測(stable.py 是多幀中位數穩定化)
├── vision/         純 CV 無狀態:roi + tiles/(hand 定位、classify 分類)
├── analysis/       shanten(向聽 / 進張)+ danger(放銃危險度,純排除法)
├── mjai/           events / tiles / handstate(自己的手牌)/ table(整桌公開資訊)
├── engine/         base / subprocess_engine / mortal / dummy / multiplex / actions
├── live/           bus / vision / packets / danger / runtime / source(開遊戲)
├── ui/             viewmodel(顯示什麼,不碰 Qt)+ state(記住的位置)
│                   + switchboard(開關的 Protocol)+ style(深色模式的顏色)
│                   + panel/(側邊視窗) overlay/(疊在遊戲上的 HUD)
│                   + widgets/(tiles, advice, analysis, danger, toggle)
├── groundtruth/    cdp / capture_addon / dump / liqi / schema / to_mjai / stream
├── eval/           align / report(準確率評測,素材待補)
└── recorder/       錄製資料集、離線回放

tools/  ui.py(側邊視窗)  gt.py(封包擷取/檢視)  advise.py(離線重播給引擎)
        evaluate.py(準確率)  record.py(畫面錄製)  roi_annotate.py  fetch_tiles.py

docs/   decisions.md  ← 方向轉折與全部決策理由(最重要)
        live.md       ← 即時模式操作手冊 + 狀態列對照表 + Overlay 怎麼用
        recording.md  ← 錄製資料集的檢查清單
        handoff.md    ← 這一份
```

### 讀原始碼的建議順序

1. `README.md` 的「架構」與「兩條資料流」
2. `docs/decisions.md` 第二、三節(為什麼是這三個功能)
3. `src/mia/ui/viewmodel.py` —— 整個 UI 要顯示什麼都在這裡,而且不碰 Qt
4. `src/mia/live/bus.py` 的模組 docstring —— 郵箱的設計理由
5. `src/mia/engine/base.py` —— 引擎介面與 `Advice` 的語意
6. `src/mia/analysis/danger.py` 的模組 docstring —— 放銃分析為什麼是排除法
   而不是機率,以及「只認立直」那個錯誤是怎麼被真實牌譜逼出來的

每個模組的 docstring 都寫了「為什麼這樣做」與「刻意不做什麼」,那是主要的
設計文件 —— 讀它們比讀程式碼快。
