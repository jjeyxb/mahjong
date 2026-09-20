# 錄一份成對的素材

一次錄製要同時滿足四件事。分開錄會拖成好幾場,而且每多一份素材就多一組
要對齊的時間軸。

| # | 目的 | 需要什麼 |
|---|---|---|
| 1 | `StableCalibrator` 端對端複驗 | 要有和了、立直、暗轉、載入畫面 —— 那些正是干擾邊界偵測的畫面 |
| 2 | 槽位幾何在第二種解析度也對 | 兩種視窗大小各錄一段 |
| 3 | per-tile 準確率報告 | 畫面與封包**同時**錄,靠牆上時鐘配對 |
| 4 | 模板庫的邊界情況 | 手牌出現過紅寶牌(`0m`/`0p`/`0s`)與白板 |

原始的 `_align_test` 錄影(216 幀 + 對應的 `ws.jsonl`)在整理專案時刪掉了,
而且屬 gitignore,無法還原。

---

## 錄之前

```bash
# 確認 CDP 那條路還通(會開一個 Chromium)
.venv/bin/python tools/gt.py cdp --duration 5 --out /tmp/probe.jsonl

# 確認擷取層找得到視窗(--list 列出所有候選)
.venv/bin/python tools/capture_probe.py --list
```

**用網頁版,不要用 Steam 桌面版。** CDP 只能接瀏覽器;桌面版要走 MITM,
而 CA 憑證那條路還沒實測過(見 `docs/decisions.md` 待辦 #6)。

---

## 錄的時候

開兩個終端機。**兩個都在登入之前就開,不要等進了牌桌才開畫面錄製。**

`record.py` 只存有變化的幀,大廳那段靜止畫面 654 幀只會存下 1 張,幾乎不佔空間。
反過來「等進對局再開」看似省事,實際上多了一個要人記得的同步點 ——
2026-07-29 第一次嘗試就是漏掉那一步,結果整場只錄到封包。

```bash
# 終端機 A —— 封包。會開一個 Chromium
.venv/bin/python tools/gt.py cdp --out data/gt/ws.jsonl --user-data-dir data/gt/browser-profile

# 終端機 B —— 畫面。瀏覽器一開好就下,不用等進牌桌
.venv/bin/python tools/record.py --window "#<handle>" --notes "評測用 東風戰"
```

`--user-data-dir` 會保留登入狀態,重錄時不用再登一次。

### 一定要用 `--window` 指定

自動偵測靠標題比對,而**開著這個專案的編輯器標題也含「雀魂」**,而且視窗更大:

```
[9594] Google Chrome for Testing — 雀魂麻將 1113x813     ← 要這個
[8072] Code — 雀魂麻將輔助軟體架構設計 — mahjong 1512x865   ← 會被誤選
```

先用 `python tools/capture_probe.py --list` 拿到 handle,再 `--window "#9594"`。

### 兩種視窗大小要錄成兩個 session

session manifest 只存**一個** `table_rect`,錄製中途縮放視窗的話,`StableCalibrator`
會重新校正(尺寸變了就重置),但 manifest 留著的仍是第一次鎖定的那個,
`tools/evaluate.py` 因此會拿舊座標去切後半段的 ROI。

所以流程是:錄一段 → 停掉 `record.py` → 縮放視窗 → 重新 `record.py`(新的 handle 要重查)。
**封包那邊不用停**,一份 `ws.jsonl` 可以同時對應兩個 session。

錄製中要做到的事:

- [ ] **打完一整場**(東風戰即可)。中途離開的話 `to_mjai` 沒有 `end_kyoku`,
      時間軸最後一段會缺。
- [ ] **中途把瀏覽器視窗縮放一次**,兩種大小各撐幾局。這是唯一能驗證
      正規化座標真的會跟著縮放的方式 —— 目前只在單一尺寸驗過。
- [ ] **至少摸到一次紅寶牌與白板**。白板是佔用判據的下限(實測 std 35.6,
      空桌面上限 23.0),紅寶牌是模板庫最容易漏掉的三張。
- [ ] **不要跳過和了動畫與局間過場**。那些畫面正是校正會出錯的地方,
      跳過就驗不到離群值過濾。

`record.py` 只存有變化的幀,所以錄久一點不會爆磁碟。

---

## 錄完立刻檢查

**趁還登入著就跑這一步。** 對不上的話重錄一場,比事後才發現省事得多。

```bash
.venv/bin/python tools/evaluate.py data/recordings/<session-id> data/gt/ws.jsonl --dry-run
```

會印出兩件事:

1. **重疊了幾秒**。是 0 或負的,代表兩份錄影根本不是同一段時間 ——
   通常是其中一邊在對局開始前就停了。
2. **不同穩定秒數下可評分的幀數**。全是 0 的話,多半是 session 沒有
   `table_rect`(錄太短,`StableCalibrator` 還沒鎖定)。

兩個數字都正常就可以跑完整評測:

```bash
.venv/bin/python tools/evaluate.py data/recordings/<session-id> data/gt/ws.jsonl
```

---

## 為什麼要「穩定秒數」

封包在動作**發生時**就到了,畫面要等動畫演完才變。摸一張牌從封包抵達到牌
真的出現在手上,中間隔著發牌動畫。所以在轉場的那半秒裡,畫面與封包必然
不一致 —— **那不是 CV 認錯**。

`eval/align.py` 因此只取穩定區間:一個狀態前後都安靜夠久,落在中間的幀才
拿來評分。代價是可用的幀變少,好處是剩下的每一幀都有明確的標準答案。

反過來做 —— 把轉場幀算進去、用一個容忍度去湊 —— 會讓準確率變成一個由容忍度
決定的數字,那就沒有意義了。

預設的 `SETTLE = 0.4` 秒是**估計值,不是量出來的**。素材錄好之後應該掃幾個值:

```bash
for s in 0.2 0.3 0.4 0.6 0.8; do
  python tools/evaluate.py <session> <dump> --settle $s | tail -5
done
```

準確率曲線平掉的那個轉折點,才是雀魂動畫的真實長度。那個數字值得寫進論文。

---

## 錄好之後要更新的地方

- `docs/decisions.md` 待辦 #1、#3 可以劃掉
- README 的「已知技術難點」表格第 4 列(校正)改成已複驗
- 準確率數字補進 README 的 M3 段落
- 挑幾幀有代表性的存進 `tests/fixtures/`(紅寶牌、白板、副露、立直橫放),
  當作永久迴歸測試 —— 整份錄影太大不進版控
