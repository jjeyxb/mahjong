"""四家牌河(棄牌區)的座標、排列方向、打牌順序。

牌河是「打了哪張牌」最可靠的觀察點:牌一打出去就攤在這裡、牌面朝上,四家
都看得見。對手的手牌是牌背朝上完全讀不出來,只有牌河讀得到。

✅ 這組座標已經驗證過(舊樹 run_step5.py 實跑,四家的框都正確涵蓋牌河,而且
依座標排出來的打牌順序與畫面上的排列一致)。數值原本是從更早的專案(1_state)
搬過來的,那邊用的是另一份錄影 —— 這種沿用一定要實際驗過才算數,舊專案就
發生過指示燈座標整組偏了 8~12px 的情況。

2026-10-23 沿用的理由與補強:框沒有改過,所以原本的驗證結論還成立。但原本的
驗證可能沒涵蓋兩種情況,所以 3_river.py 另外加了自動檢查:
    ・立直宣言牌橫放,那一排會往外長出去 —— 框夠不夠寬?
    ・被鳴走之後尾端空一格 —— 排序會不會因此亂掉?
檢查方式是「有沒有牌的框碰到裁切邊界」(碰到 = 框畫小了,而且那會**靜默**
吃掉牌,看起來只像是讀不到),加上一張疊圖讓人看一眼。
"""
from tiles import in_region

# (x, y, w, h)。範圍要涵蓋「打滿一輪之後可能累積到的最遠位置」,而且要留
# 空間給立直時橫放的那張牌多出來的寬度。
RIVER_ZONES = {
    "P0": (756, 541, 411, 214),
    "P1": (1135, 278, 263, 272),
    "P2": (788, 151, 335, 155),
    "P3": (511, 288, 265, 272),
}

# 這組座標有沒有在本專案驗證過。沿用舊數值而不標記驗證狀態,是之前踩過的
# 坑,所以這個旗標要跟上面的說明一起維護。
RIVER_ZONES_VERIFIED = True

SEATS = ("P0", "P1", "P2", "P3")


def river_of(seat):
    return RIVER_ZONES.get(seat)


def in_any_river(det, zones=None):
    zones = zones or RIVER_ZONES
    return any(in_region([det], z) for z in zones.values())


# ---------------------------------------------------------------------------
# ★★ 這張牌是哪一家牌河的牌 —— 唯一的判準是「中心點落在誰的框裡」
# ---------------------------------------------------------------------------
# 使用者 2026-10-24 重申的流程:
#   1. 模型偵測到新的牌
#   2. 多幀確認不是雜訊(RiverTracker 的投票)
#   3. 看中心點落在哪一塊固定牌河區域 -> 那就是這張牌的主人
#   4. 不在任何區域 -> 不在牌河,或是雜訊 -> 無視這張牌
#
# 為什麼要獨立寫出來:原本的歸屬是「哪一塊裁圖讀到它就算誰的」。那在目前的
# 架構下**剛好**等於中心點判斷(偵測框一定在裁圖裡面),所以 101,593 筆實測
# 資料裡兩種算法完全沒有分歧。但那是幾何的副作用,不是程式的判斷 ——
# 一旦改成整張畫面推論、框擴更大、或四家合成一個模型,那個保證就無聲消失。
# 而且框本來就有重疊角落(P0/P1 32x9px、P0/P3 20x19px),那裡現在是靠信心度
# 高低決定主人,不是靠位置。把判準寫成程式碼才擋得住這兩件事。


def zone_margin(center, zone):
    """中心點離框的四個邊界最近有多遠。負數 = 在框外面。"""
    x, y, w, h = zone
    cx, cy = center
    return min(cx - x, x + w - cx, cy - y, y + h - cy)


def seat_of_center(center, zones=None, src=None):
    """中心點落在哪一家的牌河框裡。不在任何框裡回傳 None(= 這張牌無視)。

    落在重疊角落時取「離邊界最遠」的那一家:真的在誰的牌河裡,離那一家的
    邊界就遠;只是擦到角落的那一家餘裕趨近 0。

    ★ 純幾何。**刻意不看** src(哪一塊裁圖讀到它)也不看信心度:
      ・信心度高不代表位置對
      ・同一個位置的歸屬不可以因為「這一幀是哪塊裁圖先讀到」而改變,
        不然同一張牌會在兩家之間跳來跳去,而且完全無法重現。
      餘裕剛好一樣的話照 SEATS 的順序取第一個 —— 任意但固定。
      (src 這個參數留著只為了讓呼叫端讀起來清楚,函式不會用它。)
    """
    zones = zones or RIVER_ZONES
    hits = [s for s in SEATS
            if s in zones and zone_margin(center, zones[s]) >= 0]
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    return max(hits, key=lambda s: zone_margin(center, zones[s]))


def assign_seats(dets, zones=None, center_of=lambda d: d["center"]):
    """照中心點重新決定每一筆偵測屬於哪一家。

    回傳 (保留的偵測, 改過家的, 丟掉的)。改過家的跟丟掉的要印出來 ——
    這兩件事在目前的架構下應該永遠是 0,所以一旦不是 0 就是真的出事了,
    不能安靜地吞掉。
    """
    zones = zones or RIVER_ZONES
    kept, moved, dropped = [], [], []
    for d in dets:
        src = d.get("seat")
        seat = seat_of_center(center_of(d), zones, src)
        if seat is None:
            dropped.append(d)
            continue
        if seat != src:
            moved.append((d, src, seat))
            d["seat"] = seat
        kept.append(d)
    return kept, moved, dropped


# ---------------------------------------------------------------------------
# 牌河的排列方向:用座標直接判斷打牌順序,最後一張就是剛打出的那張。
# ---------------------------------------------------------------------------
# 四家的牌河都是「一排一排」往自己的方向堆。每家的視角轉了 90/180 度,所以
# 分排的軸、排的推進方向、排內的前進方向都不一樣:
#
#   P0(下)  同一排 y 相近,排內 x 越大越新,排與排往 y 增加的方向堆
#   P1(右)  同一排 x 相近,排內 y 越小越新,排與排往 x 增加的方向堆
#   P2(上)  同一排 y 相近,排內 x 越小越新,排與排往 y 減少的方向堆
#   P3(左)  同一排 x 相近,排內 y 越大越新,排與排往 x 減少的方向堆
#
# 為什麼用座標排序而不是用「偵測到的先後順序」:
# 偵測順序會被模型閃爍影響 —— 某一幀漏看一張、下一幀又看到,順序就亂了。
# 座標是畫面上的客觀事實,每一幀都能獨立算出正確的順序,不累積誤差。
#
# 這套排序成立的前提是「牌河永遠是連續的」,而這一點成立:牌被鳴走之後那一格
# 變成空的,但那一格一定是最後一格(被鳴的一定是剛打出的那張),而且這家下一
# 次打出的牌會回到同一格。所以空位永遠在尾端、不會出現在中間,座標順序等於
# 打牌順序這件事不會因為鳴牌而破掉。
#
# (那一格的「身分」會換人 —— 第四格被吃走之後,第四格下一次出現的是另一張
#  牌。那是跨幀的事,屬於下一步,跟這裡的排序無關。)
#
# axis         分排用的軸:'y' 表示同一排的 y 相近(橫排),'x' 表示直排
# row_sign     排與排推進的方向,+1 代表該軸座標越大是越後面的排
# within_sign  同一排內,+1 代表另一軸座標越大是越新的牌
RIVER_ORDER = {
    "P0": {"axis": "y", "row_sign": +1, "within_sign": +1},
    "P1": {"axis": "x", "row_sign": +1, "within_sign": -1},
    "P2": {"axis": "y", "row_sign": -1, "within_sign": -1},
    "P3": {"axis": "x", "row_sign": -1, "within_sign": +1},
}

# 分排的容忍值。同一排的牌在分排軸上的座標會有幾 px 的抖動,不同排之間則差
# 一整個牌的高度。用「相鄰值差距超過這個數就換排」的方式分組,比固定格線穩
# —— 固定格線遇到剛好落在格線上的排會把同一排切成兩半。
ROW_TOLERANCE = 28

# 每一家的牌河「排內往哪個方向長」。直接從 RIVER_ORDER 推出來,不要另外手寫
# —— 兩份會不一致,而不一致的症狀是框往錯的方向擴,看起來像隨機漏讀。
#
# ★ 為什麼要知道這個:雀魂一排 6 張、通常 3 排,但**第 3 排可以一直往下長**。
#   一家如果一直被鳴牌不到、又打滿整局,第 3 排會超過 6 張、往排內前進的
#   方向伸出去,超出原本畫好的框。那些牌不在裁圖裡,所以連「漏讀」都看不到
#   —— 偵測結果上完全沒有痕跡。
#   (2026-10-23 使用者回報真的看過。量測方式見 3_river.py 的 --grow。)
GROW_DIR = {}
for _seat, _spec in RIVER_ORDER.items():
    if _spec["axis"] == "y":          # 排是橫的,排內沿 x 前進
        GROW_DIR[_seat] = "右" if _spec["within_sign"] > 0 else "左"
    else:                             # 排是直的,排內沿 y 前進
        GROW_DIR[_seat] = "下" if _spec["within_sign"] > 0 else "上"


# ★ 2026-11-05 量測(兩場錄影、95 組相鄰牌):P3 的第 3 排每往下一張,中心往左
#   偏 7px(透視)。而第 3 排最後一張離框的左邊只剩 3px —— 溢出的牌一定會從
#   左邊凸出去,只往下擴的框會把它切掉。往下擴 px 時,左邊跟著多擴這個比例。
#   (P1 往上長、每張往左 6px,但它的第 3 排離右邊還有 148px 以上,不用管。)
GROW_SIDE = {"P3": ("左", 0.20)}


def grow_zone(seat, zone, px):
    """把一家的框往它自己的成長方向擴 px(P3 另外往左補透視偏移)。"""
    x, y, w, h = zone
    if not px:
        return zone
    d = GROW_DIR.get(seat)
    if d == "右":
        x, y, w, h = (x, y, w + px, h)
    elif d == "左":
        x, y, w, h = (x - px, y, w + px, h)
    elif d == "下":
        x, y, w, h = (x, y, w, h + px)
    elif d == "上":
        x, y, w, h = (x, y - px, w, h + px)
    side = GROW_SIDE.get(seat)
    if side:
        e = int(round(px * side[1]))
        if side[0] == "左":
            x, w = x - e, w + e
        elif side[0] == "右":
            w = w + e
    return (x, y, w, h)


def needed_growth(seat, items, zone, lookahead=2, margin=10,
                  box_of=lambda d: d["box"], grow_from_row=2):
    """這一家的框在成長方向上還要擴多少 px,才放得下「再 lookahead 張」。

    ★★ 一定要**預先**擴,這是整個設計的重點。
       溢出框的牌根本不在裁圖裡 —— 偵測結果上一點痕跡都沒有,連「漏讀」
       都看不到。等看到被切的牌才擴,那張牌已經漏掉了。
       所以判斷是「最後一排現在有 k 張 -> 確保放得下 k+lookahead 張」。

    ★ 觸發用**幾何**不用數量。
      使用者原本的想法是「總數到 17 張就擴」,方向對,但數量判斷會被漏讀
      打敗:少讀一張就不觸發,而少讀一張正是會發生的事。改成看「最後一排
      最外面那張牌的邊緣,離框的成長邊界還有多遠」,漏讀不影響。

    ★ 一張牌多大從**那一排實際的牌**量(中位數),不寫死 —— 牌河有透視,
      越靠桌子中心的那一排畫得越大,寫死的數字對不上任何一排。

    只看**最後一排**。前面幾排固定 6 張,它們頂到邊界代表「框剛好放得下
    6 張」,那是正常的,不是溢出。

    ★ 而且只有排到第 grow_from_row+1 排(預設第 3 排)才算。
      第 1、2 排滿了之後下一張是**換排**,不是往外長 —— 在那之前擴框是
      白擴,而且會無謂地吃到鄰居的牌河。
      (2026-10-23 第一版沒有這個條件,結果每一局前六張就先擴了 118px。)

    回傳 px(0 = 夠了)。
    """
    if not items:
        return 0
    spec = RIVER_ORDER.get(seat)
    if spec is None:
        return 0
    rows = rows_of(seat, items)
    last_row = max(rows)
    if last_row < grow_from_row:
        return 0
    mine = [d for d, r in zip(items, rows) if r == last_row]
    if not mine:
        return 0
    horiz = spec["axis"] == "y"          # 排是橫的 -> 排內沿 x 前進
    # 這一排的牌沿前進方向有多寬(中位數)
    sizes = sorted((box_of(d)[2] - box_of(d)[0]) if horiz
                   else (box_of(d)[3] - box_of(d)[1]) for d in mine)
    pitch = sizes[len(sizes) // 2]
    zx, zy, zw, zh = zone
    d = GROW_DIR.get(seat)
    if d == "右":
        room = (zx + zw) - max(box_of(x)[2] for x in mine)
    elif d == "左":
        room = min(box_of(x)[0] for x in mine) - zx
    elif d == "下":
        room = (zy + zh) - max(box_of(x)[3] for x in mine)
    elif d == "上":
        room = min(box_of(x)[1] for x in mine) - zy
    else:
        return 0
    return max(0, lookahead * pitch + margin - room)


def grown_zones(px, zones=None):
    """把每一家的框往自己的成長方向擴 px。回傳新的 {seat: (x,y,w,h)}。

    只擴那一個方向 —— 四面都擴的話很容易蓋到別家的牌河,而那不是重複偵測
    (跨家去重救得了),是**認錯人**(救不了)。
    """
    zones = dict(zones or RIVER_ZONES)
    if not px:
        return zones
    return {seat: grow_zone(seat, z, px) for seat, z in zones.items()}

# 雀魂牌河一排最多幾張。非最後一排不足這個數就代表有牌沒讀到 —— 除非那一排
# 有立直宣言牌(橫放的佔比較寬,那一排會早一點換行)。
ROW_LEN = 6


def _group_rows(values):
    """把分排軸的座標分組,回傳每個值對應的排序號(從 0 開始)。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    row_of = [0] * len(values)
    row = 0
    for n, i in enumerate(order):
        if n > 0 and values[i] - values[order[n - 1]] > ROW_TOLERANCE:
            row += 1
        row_of[i] = row
    return row_of


def rows_of(seat, items, center_of=lambda d: d["center"]):
    """每個 item 屬於第幾排(0 = 離自己最近的那一排)。

    回傳的排號已經照 row_sign 正規化過,所以 0 一定是「最先打的那一排」,
    不管那一家的推進方向是往座標大的還是小的方向。
    """
    spec = RIVER_ORDER.get(seat)
    if spec is None or not items:
        return [0] * len(items)
    centers = [center_of(d) for d in items]
    perp = [c[1] if spec["axis"] == "y" else c[0] for c in centers]
    raw = _group_rows(perp)
    top = max(raw) if raw else 0
    return raw if spec["row_sign"] > 0 else [top - r for r in raw]


def sort_by_discard_order(seat, items, center_of=lambda d: d["center"]):
    """把一家牌河裡的牌依「打出的先後」排好,最後一個就是最新打出的那張。

    items 可以是任何有中心點的物件,用 center_of 取出 (x, y)。
    """
    spec = RIVER_ORDER.get(seat)
    if spec is None or not items:
        return list(items)

    centers = [center_of(d) for d in items]
    perp = [c[1] if spec["axis"] == "y" else c[0] for c in centers]
    along = [c[0] if spec["axis"] == "y" else c[1] for c in centers]
    rows = _group_rows(perp)

    return [items[i] for i in sorted(
        range(len(items)),
        key=lambda i: (spec["row_sign"] * rows[i],
                       spec["within_sign"] * along[i]))]


def newest(seat, items, center_of=lambda d: d["center"]):
    """回傳這家牌河裡最新打出的那一張,沒有牌回傳 None。"""
    ordered = sort_by_discard_order(seat, items, center_of)
    return ordered[-1] if ordered else None


# ---------------------------------------------------------------------------
# 橫放的那一張(立直宣言牌)
# ---------------------------------------------------------------------------
# 為什麼不能用固定的長寬比判斷:P1/P3 的牌河是從側面看的,那裡**正常**的牌
# 本來就跟 P0 正常的牌長寬比不一樣。所以判準只能是相對的 —— 跟**同一家
# 這一幀其他牌**的長寬比比較,反過來的那一張就是橫放的。
#
# 這樣寫的另一個好處是自我校準:遊戲改版、視窗比例變了都不用改這裡的數字。
SIDEWAYS_RATIO = 1.6      # 跟中位數差這麼多倍才算「反過來」


def _ratio(box):
    x1, y1, x2, y2 = box
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    return w / float(h)


def _median(vals):
    v = sorted(vals)
    return v[len(v) // 2] if v else 1.0


def mark_sideways(items, box_of=lambda d: d["box"], ratio=SIDEWAYS_RATIO):
    """哪幾張是橫放的。回傳跟 items 等長的 bool 清單。

    少於 3 張就全部回 False —— 樣本太少算不出可靠的中位數,而「寧可說不知道
    也不要猜」在這個專案裡已經是定好的原則了。
    """
    if len(items) < 3:
        return [False] * len(items)
    rs = [_ratio(box_of(d)) for d in items]
    mid = _median(rs)
    if mid <= 0:
        return [False] * len(items)
    return [(r / mid >= ratio) or (mid / r >= ratio) for r in rs]
