#!/usr/bin/env python3
"""摸切機器人 —— **只為了錄素材,不是 app 的功能。**

為什麼存在
----------
量牌河 CV 的準確率需要「畫面與封包涵蓋同一段時間、而且有多家立直」的素材,
而那表示要有人坐在那裡打完整局。實測上一份素材就是這樣壞掉的:
``20260801-110639`` 的畫面只錄 848 秒、封包 1433 秒,那一局 7 次立直只涵蓋
到**一家**,量「立直前 vs 立直後」時幅度就釘不下來。

所以這支程式替你坐在那裡。**只用在對電腦的房間** —— 在真人對局裡放機器人
影響的是每一局另外三個人,那是另一回事。

兩種機器人
----------
``TsumogiriBot`` 摸到什麼打什麼。簡單、不會錯,但打出來的牌局**不像真的**:
自己這家永遠不立直、不鳴牌,而那會連帶改變三家電腦的行為(沒人立直就沒人
退守)。拿它錄出來的素材,立直的分布與真實對局不一樣。

``MortalBot`` 接真正的引擎。慢、要載權重、而且多了一整類失敗模式(決策對
但按錯地方),換來的是**局面像真的** —— 自己會立直、會鳴牌,別家會因此退守。
要量的東西(立直牌在畫面上長什麼樣、牌河怎麼長)正好是被這些行為決定的。

兩個都留著:摸切是「先確認點得到」的那一步,Mortal 是正式錄素材用的。

它怎麼知道輪到自己
------------------
不靠畫面 —— 靠封包。:class:`~mia.mjai.handstate.HandTracker` 的 ``drawn``
不是 ``None`` 就代表「這一巡摸到了牌、還沒打」。那是精確的,而且是既有的
程式碼,不是為這支工具新寫的判斷。

用法::

    python tools/gt.py cdp --autoplay --out data/recordings/x/ws.jsonl
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from mia.groundtruth.dump import DumpFrame
from mia.groundtruth.stream import MjaiDecoder
from mia.mjai.handstate import HandTracker
from mia.mjai.tiles import sort_key
from mia.utils.logging import logger

if TYPE_CHECKING:  # pragma: no cover
    from playwright.sync_api import Page

__all__ = [
    "DRAWN_TILE",
    "SHOT_DIR",
    "MortalBot",
    "TsumogiriBot",
    "tile_point",
    "viewport",
]

#: dry-run 時畫面存去哪。
SHOT_DIR = Path("out/autoplay")

#: 心跳間隔(秒)。
_HEARTBEAT = 10.0

#: 切牌最多存幾張。
#:
#: 存切牌不只是為了對牌的座標 —— **立直鈕會出現在這些圖裡**。立直是在
#: 自己的回合可以宣告的,而切牌也是自己的回合,所以切牌前的畫面上只要
#: 這一巡能立直,那顆按鈕就在。這是拿到立直座標最省事的辦法:不必等引擎
#: 真的決定立直(實測整場 81 個決策裡只有 2 次)。
#:
#: 20 張約 56 MB,錄一場夠用。按鈕本身不設上限 —— 那幾顆難得跳一次。
_TILE_SHOTS = 20

#: 剛摸進來那張牌在畫面上的位置,相對 viewport 的比例 (x, y)。
#:
#: 來源是 ``docs/decisions.md`` 第四節量過的手牌 ROI:整理好的 13 張到
#: x=0.757,摸的那張空一格後落在 [0.774, 0.822] —— 取中點 0.798;
#: y 取 own_hand ROI 的垂直中點 0.853 + 0.144/2 = 0.925。
#:
#: 那組座標是相對**牌桌矩形**量的,這裡當成相對 viewport 用。兩者不完全
#: 相等(實測長寬比 1.773 vs 1.778),但一張牌佔 4.8% 寬,容許誤差 2.4%,
#: 而兩者的差距遠小於此。按錯的後果也只是這一巡沒打出去、下一巡再試。
DRAWN_TILE = (0.798, 0.925)

#: 兩次點擊至少間隔幾秒。
#:
#: 不是為了像人 —— 是因為**封包與畫面有延遲**。打出去之後 ``drawn`` 要等
#: 伺服器回的那個事件才會清掉,在那之前狀態仍然說「該打牌」。沒有這道閘
#: 就會連點,而連點在雀魂上的後果是把下一巡的牌也打掉。
_COOLDOWN = 2.0


def viewport(page: Page) -> tuple[int, int] | None:
    """分頁的可視區大小。拿不到回 ``None``。

    **不能只看 ``page.viewport_size``。** 瀏覽器是用 ``no_viewport=True``
    開的(那是為了讓雀魂跟著視窗縮放重新排版,見 cdp.py),而那個選項的
    意思就是「不要設 viewport」—— 於是這個屬性永遠是 ``None``。

    實測這正是「自動打牌完全沒反應」的原因:計畫做好了、在等著動手,
    而 ``act`` 每次都卡在 ``if not size: return`` 然後**安靜地**返回。
    心跳印出「等著動手」但「動作 0」才看得出來。

    所以直接問頁面自己。``window.innerWidth`` 不管有沒有設 viewport 都對,
    而且它就是 CSS 像素 —— 與 ``page.mouse`` 吃的座標系同一個。
    """
    try:
        size = page.evaluate("() => [window.innerWidth, window.innerHeight]")
    except Exception as exc:  # noqa: BLE001 - 分頁可能剛被關掉
        logger.warning("問不到可視區大小:{}", exc)
        return None
    if not size or len(size) != 2 or not all(isinstance(v, int | float) and v > 0 for v in size):
        return None
    return int(size[0]), int(size[1])


#: 按住多久(毫秒)。
#:
#: **不能用 ``page.mouse.click``。** 它的 down 與 up 之間幾乎沒有間隔,
#: 而雀魂是畫在 canvas 上的遊戲 —— 這類遊戲多半是每個繪製影格去**輪詢**
#: 輸入狀態,而不是監聽事件。零長度的按壓有機會整個落在兩個影格之間,
#: 於是遊戲從頭到尾沒看到「現在是按下的」。
#:
#: 120 ms 在 60 fps 下橫跨約 7 個影格,怎麼樣都會被輪詢到。
_PRESS_MS = 120


def _press(page: Page, x: int, y: int) -> None:
    """在 (x, y) 按一下 —— 移過去、按住、放開。

    先 ``move`` 再按:canvas 遊戲常常要先收到 hover 才會把那個位置當成
    可互動的目標(按鈕的高亮狀態就是這樣來的)。直接在沒有 hover 的位置
    按下去,有些實作會當作沒按在任何東西上。
    """
    page.mouse.move(x, y)
    page.wait_for_timeout(60)
    page.mouse.down()
    page.wait_for_timeout(_PRESS_MS)
    page.mouse.up()


@dataclass
class TsumogiriBot:
    """摸到什麼打什麼。滿足 :class:`~mia.groundtruth.cdp.AutoPlayer`。"""

    #: 打出去之前先等幾秒。給畫面把摸牌動畫演完 —— 動畫中間點下去不會生效。
    delay: float = 1.0

    _decoder: MjaiDecoder = field(default_factory=MjaiDecoder, repr=False)
    _hand: HandTracker = field(default_factory=HandTracker, repr=False)
    _ready_at: float | None = field(default=None, repr=False)
    _last_click: float = field(default=0.0, repr=False)

    #: 統計,給人看的。
    discards: int = 0

    def observe(self, payload: bytes, *, from_client: bool, flow: str) -> None:
        frame = DumpFrame(
            timestamp=0.0, wall_clock=time.time(), from_client=from_client,
            payload=payload, flow=flow,
        )
        decoded = self._decoder.feed(frame)
        if decoded is None:
            return
        for event in decoded.events:
            if event.TYPE == "start_game":
                # 新的一場:座位可能變了,追蹤器要從頭來。
                self._hand = HandTracker(_name="autoplay")
            self._hand.feed(event)
        self._ready_at = time.monotonic() + self.delay if self._hand.drawn else None

    def act(self, page: Page) -> None:
        now = time.monotonic()
        if self._ready_at is None or now < self._ready_at:
            return
        if now - self._last_click < _COOLDOWN:
            return
        size = viewport(page)
        if size is None:
            logger.warning("拿不到可視區大小,這一巡跳過")
            return
        x = int(size[0] * DRAWN_TILE[0])
        y = int(size[1] * DRAWN_TILE[1])
        logger.info("摸切 {} —— 點 ({}, {})", self._hand.drawn, x, y)
        page.mouse.click(x, y)
        self._last_click = now
        self._ready_at = None
        self.discards += 1


# --------------------------------------------------------------------- 按哪裡

#: ``own_hand`` ROI,相對牌桌矩形 —— 與 ``config/default.yaml`` 同一組數字。
#: 寫死在這裡而不是讀設定:這支工具不是 app 的一部分,不該在使用者改了
#: 設定之後跟著動。對不上的時候該改的是這裡,而且會看得出來。
_HAND_ROI = (0.1150, 0.8533, 0.7083, 0.1437)


def tile_point(index: int, concealed: int, *, drawn: bool = False) -> tuple[float, float]:
    """第 ``index`` 張牌在畫面上的位置,相對 viewport 的比例。

    幾何來自 :class:`~mia.vision.tiles.hand.HandLayout` —— 那是量過的
    (13 張素材線性擬合,殘差 < 1 px,而且掃過畫布 480~3840)。這裡只是把
    「ROI 內的比例」換算成「viewport 內的比例」。

    Args:
        index: 第幾張暗手牌。``drawn`` 為真時忽略。
        concealed: 暗手牌有幾張 —— 決定摸的那張擺在哪。
        drawn: 要的是剛摸進來那張。

    Note:
        ROI 是相對**牌桌矩形**量的,這裡當成相對 viewport。兩者不相等
        (實測長寬比 1.773 vs 1.778),但一張牌佔 4.8% 寬,而差距遠小於此。
    """
    from mia.vision.tiles.hand import HandLayout

    layout = HandLayout()
    rx, ry, rw, rh = _HAND_ROI
    if drawn:
        inner = layout.origin + concealed * layout.pitch + layout.draw_gap
    else:
        inner = layout.origin + index * layout.pitch
    return rx + (inner + layout.width / 2) * rw, ry + rh / 2


#: 按鈕那一排的垂直位置與水平間距,相對 viewport。**量過的**:
#: 2026-10-24 從 ``out/autoplay/skip-01`` 量,碰 (0.533, 0.760)、
#: 跳過 (0.682, 0.760),兩者相距 0.149。兩個點都確認正中按鈕。
_BUTTON_Y = 0.760
_BUTTON_PITCH = 0.149

#: 最右邊那一格的位置。雀魂把按鈕**靠右對齊**,而「跳過」永遠在最右邊 ——
#: 所以固定座標是錯的模型:按鈕數量一變,其他每一顆都跟著移位。
_BUTTON_RIGHT = 0.682

#: 從右邊數,每一種按鈕在第幾格。
#:
#: ⚠ **只有 0 與 1 是量過的。** 跳過在第 0 格、碰在第 1 格,兩者都用
#: 實際畫面確認過。其餘是推測:同時出現兩個以上可鳴的動作本來就少見,
#: 而最常見的情況(只有一種可鳴 + 跳過)不管是哪一種都落在第 1 格 ——
#: 那一格已經驗過了。
#:
#: 立直也在這一排,**與碰同一格**。使用者實際看過畫面這麼回報的。
#: 它出現在自己的回合(那時候沒有「跳過」),所以原本以為它不在這一排、
#: 要另外量 —— 不必。
_BUTTON_SLOT = {"skip": 0, "pon": 1, "chi": 1, "kan": 1, "hora": 1, "reach": 1}


def button_point(name: str) -> tuple[float, float]:
    """某一顆按鈕在畫面上的位置,相對 viewport 的比例。"""
    slot = _BUTTON_SLOT.get(name, 1)
    return _BUTTON_RIGHT - slot * _BUTTON_PITCH, _BUTTON_Y


_BUTTONS: dict[str, tuple[float, float]] = {name: button_point(name) for name in _BUTTON_SLOT}


@dataclass
class MortalBot:
    """照引擎的建議打。滿足 :class:`~mia.groundtruth.cdp.AutoPlayer`。

    與 :class:`TsumogiriBot` 的差別只在「打哪一張」—— 看封包決定輪到誰、
    冷卻時間、例外處理那些完全一樣,所以那部分是共用的。
    """

    engine: object  #: :class:`~mia.engine.base.AIEngine`
    delay: float = 1.0
    dry_run: bool = False

    _decoder: MjaiDecoder = field(default_factory=MjaiDecoder, repr=False)
    _hand: HandTracker = field(default_factory=HandTracker, repr=False)
    _pending: list[tuple[float, float]] = field(default_factory=list, repr=False)
    _ready_at: float | None = field(default=None, repr=False)
    _last_click: float = field(default=0.0, repr=False)

    actions: int = 0
    #: 看過幾個封包、解出幾個事件。心跳會報,用來分辨「沒在試」與「試了沒用」。
    frames: int = 0
    events: int = 0
    _shots: int = field(default=0, repr=False)
    _beat_at: float = field(default=0.0, repr=False)
    _tile_shots: int = field(default=0, repr=False)

    def observe(self, payload: bytes, *, from_client: bool, flow: str) -> None:
        frame = DumpFrame(
            timestamp=0.0, wall_clock=time.time(), from_client=from_client,
            payload=payload, flow=flow,
        )
        self.frames += 1
        decoded = self._decoder.feed(frame)
        if decoded is None:
            return
        for event in decoded.events:
            self.events += 1
            if event.TYPE == "start_game":
                self._hand = HandTracker(_name="autoplay")
            self._hand.feed(event)
            advice = self.engine.react(event)  # type: ignore[attr-defined]
            points = self._plan(self._resolve(advice))
            if points:
                self._pending = points
                self._ready_at = time.monotonic() + self.delay

    def _resolve(self, advice):
        """立直的話,先問清楚宣言之後要切哪一張。

        ``react()`` **不會**填 ``follow_up`` —— 那是 ``EngineGroup`` 另外
        ``peek`` 一次填的(見 ``mia/engine/multiplex.py``)。這支程式直接用
        ``react()``,所以得自己問。

        不問的後果是**安靜的**:按下立直鈕之後不選牌,雀魂等到超時替你打
        一張。對局照樣繼續、素材照樣錄下來,只是宣言牌變成系統挑的 ——
        而畫面上那張橫放的牌正是我們要量的東西。

        實測餵真實牌譜時立直出現 2 次,而 ``follow_up`` 兩次都是 None。
        單元測試抓不到這個,因為它自己捏了一個 follow_up 餵進去。
        """
        from dataclasses import replace

        from mia.engine.base import EngineError
        from mia.mjai.events import Reach

        action = getattr(advice, "action", None)
        if action is None or action.TYPE != "reach":
            return advice
        actor = int(getattr(action, "actor", 0))
        try:
            follow = self.engine.peek(Reach(actor=actor))
        except EngineError as exc:
            logger.warning("問不到立直後要切哪張:{}", exc)
            return advice
        return replace(advice, follow_up=follow.action)

    def _plan(self, advice) -> list[tuple[float, float]]:
        """把一個建議翻成「要點哪幾個地方」。空清單代表不必動。

        **這個函式不碰畫面也不碰時間**,所以它是可以測的 —— 而它正是這支
        程式唯一會「想錯」的地方。
        """
        action = getattr(advice, "action", None)
        if action is None:
            # 被問了但選擇跳過。沒被問的話 declined 是 False,就什麼都不做。
            return [_BUTTONS["skip"]] if getattr(advice, "declined", False) else []

        kind = action.TYPE
        if kind == "dahai":
            point = self._tile_point(action.pai)
            return [point] if point else []
        if kind == "reach":
            # 立直是兩段:先宣言,再切。切哪一張引擎先 peek 好了放在 follow_up
            # —— 等真的看到 reach 事件才問就來不及了(見 Advice.follow_up)。
            points = [_BUTTONS["reach"]]
            follow = getattr(advice, "follow_up", None)
            if follow is not None and getattr(follow, "TYPE", "") == "dahai":
                point = self._tile_point(follow.pai)
                if point:
                    points.append(point)
            return points
        if kind in _BUTTONS:
            return [_BUTTONS[kind]]
        logger.info("不知道怎麼按「{}」,跳過這一手", kind)
        return []

    def _tile_point(self, pai: str) -> tuple[float, float] | None:
        """手上哪一張是這張牌 —— 用封包的手牌找,不用畫面。

        雀魂把暗手牌排序後顯示,摸進來那張另外擺在右邊。所以位置由
        「它在排序後的第幾個」決定,而那個順序封包就知道。
        """
        # **只移掉一張,不是所有同名的。** 摸到一張手上已經有的牌很常見,
        # 而按值過濾會把兩張都拿掉 —— 手牌數少算一張,摸的那張就往左偏
        # 一格,點到隔壁。隔壁那張是合法的牌,所以不會有任何錯誤訊息,
        # 只是打錯牌。
        rest = list(self._hand.tiles)
        if self._hand.drawn is not None and self._hand.drawn in rest:
            rest.remove(self._hand.drawn)
        if self._hand.drawn == pai:
            return tile_point(0, len(rest), drawn=True)
        # **一定要用 sort_key,不能用 sorted() 直接排字串。** 字串序是
        # 先比點數再比花色,排出來是 1m 1p 1s 2m 2p 2s…,而雀魂是按花色
        # 分組的;字牌更離譜,字母序排成 C E F N P S W 而不是東南西北白發中。
        # 症狀是「有時候按歪」—— 而且只要手上牌的花色一混就會歪。
        order = sorted(rest, key=sort_key)
        if pai not in order:
            logger.warning("引擎要打 {},但手上沒有 —— 手牌 {}", pai, order)
            return None
        return tile_point(order.index(pai), len(rest))

    def act(self, page: Page) -> None:
        now = time.monotonic()
        self._heartbeat(now)
        if self._ready_at is None or now < self._ready_at or not self._pending:
            return
        if now - self._last_click < _COOLDOWN:
            return
        size = viewport(page)
        if size is None:
            logger.warning("拿不到可視區大小,這一巡跳過")
            return
        # 一次只做一步。立直是「宣言 + 切牌」兩步,中間要等畫面換 ——
        # 連著點下去第二下會落在還沒出現的按鈕上。
        fx, fy = self._pending.pop(0)
        x, y = int(size[0] * fx), int(size[1] * fy)
        # **不管有沒有真的點都要印。** 原本只有 dry-run 會出聲,於是
        # 「沒反應」分不出是「沒在試」還是「試了但遊戲沒收到」—— 使用者
        # 回報「沒有自動打牌」的時候,我手上一點資訊都沒有。
        logger.info(
            "{}點 ({}, {}) —— 比例 ({:.3f}, {:.3f})",
            "[dry-run] 會" if self.dry_run else "",
            x, y, fx, fy,
        )
        # **按鈕一律先存畫面,就算不是 dry-run。** 那幾顆的座標是估的,
        # 而它們難得跳一次 —— 錯過一次就要再等一局。切牌不存(座標有量測
        # 基礎,而且一局七十幾次)。
        name = next((k for k, v in _BUTTONS.items() if v == (fx, fy)), None)
        if name is not None or self.dry_run:
            self._snapshot(page, fx, fy)
        if not self.dry_run:
            _press(page, x, y)
        self._last_click = now
        self.actions += 1
        if not self._pending:
            self._ready_at = None

    def _heartbeat(self, now: float) -> None:
        """每隔一段時間報一次自己在哪一步。

        **「什麼都沒發生」是最難查的回報**,因為它同時符合好幾種原因:
        封包沒進來、事件解不開、引擎沒被問、輪到自己但冷卻中、或是點了
        但遊戲沒收到。這一行把那幾種分開 —— 每一個數字對應一個環節,
        卡在哪一環,哪個數字就不會動。
        """
        if now - self._beat_at < _HEARTBEAT:
            return
        self._beat_at = now
        logger.info(
            "自動打牌:封包 {} / 事件 {} / 動作 {} / 手牌 {} 張,摸到 {} / {}",
            self.frames,
            self.events,
            self.actions,
            len(self._hand.tiles),
            self._hand.drawn or "—",
            "等著動手" if self._pending else "沒輪到",
        )

    def _snapshot(self, page: Page, fx: float, fy: float) -> None:
        """dry-run 時把當下的畫面存下來,檔名帶著要點的位置。

        **只存按鈕,不存切牌。** 牌的座標是算出來的(``HandLayout`` 量過,
        殘差 < 1 px),不需要用眼睛確認;按鈕的座標是**估的**,只能靠對照
        畫面。每一巡都存會淹掉有用的那幾張。

        用 ``page.screenshot`` 而不是螢幕擷取:它拍的就是 viewport,
        與我們點擊的座標系**完全同一個**。拿螢幕擷取來對的話,還要再處理
        視窗邊框與 Retina 倍率,那又是兩個可能對錯的地方。
        """
        name = next((k for k, v in _BUTTONS.items() if v == (fx, fy)), None)
        if name is None:
            # 切牌。牌的座標有量測基礎,但那是**推算到 viewport** 的推算 ——
            # 中間差一個「牌桌矩形 ≈ viewport」的近似,沒有人驗過。所以也存
            # 幾張,但只存前面幾次:一局七十幾次切牌,每次都存會淹掉按鈕那幾張。
            if self._tile_shots >= _TILE_SHOTS:
                return
            self._tile_shots += 1
            name = "tile"
        self._shots += 1
        path = SHOT_DIR / f"{name}-{self._shots:02d}-{fx:.3f}x{fy:.3f}.png"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(path))
            logger.info("  畫面存到 {} —— 拿它對座標", path)
        except Exception as exc:  # noqa: BLE001 - 存不下來不值得中斷
            logger.warning("存不下畫面:{}", exc)
