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

只動全電腦的桌
--------------
「只在對電腦的房間用」原本只是一句提醒。現在兩個機器人都要先看到 ``authGame``
說另外三家都是電腦才會按(:func:`cpu_table`),沒看到或有真人就一動也不動。

用法::

    python tools/gt.py cdp --autoplay mortal --mortal models/x.pth --cpu-room
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from mia.groundtruth.dump import DumpFrame
from mia.groundtruth.liqi import LiqiMessage, LiqiParseError, LiqiParser, MessageKind
from mia.groundtruth.schema import LiqiSchema
from mia.groundtruth.stream import MjaiDecoder
from mia.mjai.handstate import HandTracker
from mia.mjai.tiles import ms_to_mjai, sort_key
from mia.utils.logging import logger

if TYPE_CHECKING:  # pragma: no cover
    from playwright.sync_api import Page

__all__ = [
    "DRAWN_TILE",
    "SHOT_DIR",
    "CpuRoomPilot",
    "MortalBot",
    "TsumogiriBot",
    "choice_point",
    "cpu_table",
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

#: 莊家開局要等幾秒才點。
#:
#: 閒家第一手來的時候發牌早就演完了,莊家的第一手卻在開局封包就到。逐幀看:+1 秒牌還在發,
#: +2 秒排好,+3 秒起點得到。點在動畫中間的那一下會被吃掉。
_DEAL_DELAY = 4.0

#: 點下去之後這麼久都沒有新的事件,就當作沒點到,再點一次。
#:
#: 會需要這個是因為**莊家開局**:閒家的第一手來的時候發牌動畫早就演完了,莊家的第一手卻在
#: 開局封包就到,一秒後點下去還在發牌,點擊被吃掉。實測閒家第一手 1.4~1.6 秒,莊家 13.9 秒
#: 與 53.7 秒,其他手中位 1.4 秒。
#:
#: 開局動畫的等待見 :data:`_DEAL_DELAY`,這裡是動畫之外的保險。重點是安全的:打出去之後
#: 新的事件會清掉這件事,而輪到自己之前一定先有事件。
_RETRY_AFTER = 2.5

#: 最多重點幾次。再點不到就是別的問題了,不要一直點下去。
_MAX_RETRIES = 8

_AUTH_GAME = ".lq.FastTest.authGame"

#: 四人桌要有幾台電腦。
_CPU_SEATS = 3


def cpu_table(message: LiqiMessage) -> bool | None:
    """這則訊息說這一桌是不是只有自己一個真人。與這件事無關的訊息回 ``None``。

    ``authGame`` 的回應列出 ``seat_list``(每個座位的帳號)與 ``robots``(電腦的帳號)。
    電腦的帳號是 1、2、3 這種小數字,不是 0,所以要對 ``robots``,不能看座位是不是空的。

    友人房的 ``mode.ai`` 不能當判據:實測沒按「人機」也是 ``true``,那只代表這間房可以加
    電腦,不代表桌上全是電腦。
    """
    if message.method != _AUTH_GAME or message.kind is not MessageKind.RESPONSE:
        return None
    payload: Any = message.payload
    robots = {robot.account_id for robot in payload.robots}
    humans = [account for account in payload.seat_list if account not in robots]
    return len(robots) == _CPU_SEATS and len(humans) == 1


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


def _read_operations(action) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """動作訊息附帶的選項,不含打牌。沒有問自己的話是空的。"""
    operation = getattr(action, "operation", None)
    if not operation:
        return ()
    return tuple(
        (item.type, tuple(item.combination))
        for item in operation.operation_list
        if item.type != _OP_DISCARD
    )


def _discard_of(advice) -> str | None:
    """這個建議最後會打哪張牌。立直的話是宣言之後切的那張。"""
    action = getattr(advice, "action", None)
    if action is None:
        return None
    if action.TYPE == "reach":
        action = getattr(advice, "follow_up", None)
    return action.pai if action is not None and action.TYPE == "dahai" else None


def _judge_table(message: LiqiMessage, allowed: bool) -> bool:
    """看一則訊息之後還能不能動手。不是 ``authGame`` 就維持原樣。"""
    verdict = cpu_table(message)
    if verdict is None:
        return allowed
    if verdict:
        logger.info("確認這一桌另外三家都是電腦")
    else:
        logger.warning("這一桌有其他真人 —— 自動打牌不會動手")
    return verdict


@dataclass
class TsumogiriBot:
    """摸到什麼打什麼。滿足 :class:`~mia.groundtruth.cdp.AutoPlayer`。"""

    #: 打出去之前先等幾秒。給畫面把摸牌動畫演完 —— 動畫中間點下去不會生效。
    delay: float = 1.0

    _decoder: MjaiDecoder = field(default_factory=MjaiDecoder, repr=False)
    _hand: HandTracker = field(default_factory=HandTracker, repr=False)
    _ready_at: float | None = field(default=None, repr=False)
    _last_click: float = field(default=0.0, repr=False)
    #: 這一桌確定只有電腦。見 :func:`cpu_table`。
    allowed: bool = False

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
        self.allowed = _judge_table(decoded.message, self.allowed)
        for event in decoded.events:
            if event.TYPE == "start_game":
                # 新的一場:座位可能變了,追蹤器要從頭來。
                self._hand = HandTracker(_name="autoplay")
            self._hand.feed(event)
        self._ready_at = time.monotonic() + self.delay if self._hand.drawn else None

    def act(self, page: Page) -> None:
        now = time.monotonic()
        if not self.allowed or self._ready_at is None or now < self._ready_at:
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

#: 只有一種動作可選時,每一種按鈕在第幾格,從右邊數。跳過在第 0 格,其他都在第 1 格。
#:
#: 立直也在這一排,**與碰同一格**。它出現在自己的回合,那時候沒有「跳過」。
_BUTTON_SLOT = {"skip": 0, "pon": 1, "chi": 1, "kan": 1, "hora": 1, "reach": 1}

#: 封包 ``operation_list`` 的動作代號,依引擎的動作名稱。
#:
#: 同時有好幾種動作時,按鈕由左到右就是這份清單的順序,跳過在最右邊。實測三種組合:
#: 吃 碰 跳過、碰 和 跳過、碰 槓 跳過。
_OPERATIONS = {
    "chi": (2,),
    "pon": (3,),
    "ankan": (4,),
    "daiminkan": (5,),
    "kakan": (6,),
    "reach": (7,),
    "hora": (8, 9),
}

#: 封包裡「打牌」的代號。它不是按鈕。
_OP_DISCARD = 1

#: 吃牌有好幾種組合時跳出來的選擇框,相對 viewport 的比例。
#:
#: 2026-11-01 從兩組與三組的畫面量的:組與組等距,以畫面中央排開,由左到右就是封包
#: ``combination`` 的順序。
_CHOICE_CENTER = 0.4895
_CHOICE_PITCH = 0.104
_CHOICE_Y = 0.71


def button_point(name: str) -> tuple[float, float]:
    """只有一種動作可選時,某一顆按鈕的位置,相對 viewport 的比例。"""
    return _slot_point(_BUTTON_SLOT.get(name, 1))


def _slot_point(slot: int) -> tuple[float, float]:
    return _BUTTON_RIGHT - slot * _BUTTON_PITCH, _BUTTON_Y


def choice_point(index: int, count: int) -> tuple[float, float]:
    """選擇框裡第 ``index`` 組的位置,共 ``count`` 組。"""
    return _CHOICE_CENTER + (index - (count - 1) / 2) * _CHOICE_PITCH, _CHOICE_Y


def _choice_index(consumed: Sequence[str], combinations: Sequence[str]) -> int | None:
    """引擎要的組合是封包列的第幾組。``combination`` 長得像 ``1s|2s``,赤五寫成 ``0p``。"""
    wanted = sorted(consumed)
    for index, combination in enumerate(combinations):
        if sorted(ms_to_mjai(tile) for tile in combination.split("|")) == wanted:
            return index
    return None


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
    #: 這一桌確定只有電腦。見 :func:`cpu_table`。
    allowed: bool = False
    #: 剛點的那一下,還沒看到結果。見 :data:`_RETRY_AFTER`。
    _unconfirmed: tuple[float, float] | None = field(default=None, repr=False)
    _retry_at: float = field(default=0.0, repr=False)
    _retries: int = field(default=0, repr=False)
    _seat: int | None = field(default=None, repr=False)
    #: 自己是莊家,而且還沒動過。見 :meth:`_tile_point`。
    _dealing: bool = field(default=False, repr=False)
    #: 這一手要打的牌。與打出去的不同就是點錯了。
    _intended: str | None = field(default=None, repr=False)
    #: 這一手要鳴的組合。
    _wanted_call: list[str] | None = field(default=None, repr=False)
    #: 伺服器說現在能做什麼:``(代號, 組合)``,順序就是畫面上按鈕的順序。
    _operations: tuple[tuple[int, tuple[str, ...]], ...] = field(default=(), repr=False)

    actions: int = 0
    #: 打出去的牌與引擎要的不同幾次。不是零就代表座標算錯了。
    misplays: int = 0
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
        self.allowed = _judge_table(decoded.message, self.allowed)
        if decoded.message.is_action:
            self._operations = _read_operations(decoded.message.action)
        for event in decoded.events:
            self.events += 1
            self._unconfirmed = None
            self._follow(event)
            self._hand.feed(event)
            advice = self._resolve(self.engine.react(event))  # type: ignore[attr-defined]
            points = self._plan(advice)
            if points:
                self._pending = points
                wait = _DEAL_DELAY if self._dealing else self.delay
                self._ready_at = time.monotonic() + wait
                self._intended = _discard_of(advice)
                consumed = getattr(getattr(advice, "action", None), "consumed", None)
                self._wanted_call = sorted(consumed) if consumed else None

    def _follow(self, event) -> None:
        """記下輪到誰、開局的狀態,順便核對上一手打出去的是不是要打的那張。"""
        if event.TYPE == "start_game":
            self._hand = HandTracker(_name="autoplay")
            self._seat = event.id
        elif event.TYPE == "start_kyoku":
            self._dealing = event.oya == self._seat
            self._intended = None
        elif getattr(event, "actor", None) == self._seat and event.TYPE != "tsumo":
            self._dealing = False
            if event.TYPE == "dahai":
                if self._intended not in (None, event.pai):
                    self.misplays += 1
                    logger.warning("點錯了:要打 {},打出去的是 {}", self._intended, event.pai)
                self._intended = None
            elif event.TYPE in ("chi", "pon") and self._wanted_call is not None:
                if sorted(event.consumed) != self._wanted_call:
                    self.misplays += 1
                    logger.warning(
                        "點錯了:要鳴 {},鳴到的是 {}", self._wanted_call, event.consumed
                    )
                self._wanted_call = None

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
            points = self._call_points(kind, action)
            follow = getattr(advice, "follow_up", None)
            if follow is not None and getattr(follow, "TYPE", "") == "dahai":
                point = self._tile_point(follow.pai)
                if point:
                    points.append(point)
            return points
        if kind in _OPERATIONS:
            return self._call_points(kind, action)
        logger.info("不知道怎麼按「{}」,跳過這一手", kind)
        return []

    def _call_points(self, kind: str, action) -> list[tuple[float, float]]:
        """按鈕,以及有好幾種組合時選擇框裡的那一組。

        只按按鈕的話選擇框就開在那裡,要等重點剛好點中它,而那個位置落在框的中間偏右:
        實測東三局引擎要吃 1 2 索,三組裡吃到的是中間的 2 4 索。
        """
        codes = [code for code, _ in self._operations]
        code = next((c for c in _OPERATIONS[kind] if c in codes), None)
        if code is None:
            return [button_point("kan" if kind in ("ankan", "daiminkan", "kakan") else kind)]
        points = [_slot_point(len(codes) - codes.index(code))]
        combinations = dict(self._operations)[code]
        consumed = getattr(action, "consumed", None)
        if kind in ("chi", "pon") and consumed and len(combinations) > 1:
            index = _choice_index(consumed, combinations)
            if index is None:
                logger.warning("{} 不在選項 {} 裡", consumed, combinations)
            else:
                points.append(choice_point(index, len(combinations)))
        return points

    def _tile_point(self, pai: str) -> tuple[float, float] | None:
        """手上哪一張是這張牌 —— 用封包的手牌找,不用畫面。

        雀魂把暗手牌排序後顯示,摸進來那張另外擺在右邊。所以位置由
        「它在排序後的第幾個」決定,而那個順序封包就知道。
        """
        if self._dealing:
            return self._dealt_point(pai)
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

    def _dealt_point(self, pai: str) -> tuple[float, float] | None:
        """莊家開局的 14 張在哪裡。**與封包說的摸牌無關。**

        封包的第 14 張是「摸的」,畫面不是這樣擺的:14 張一起排序,最後一張放在摸牌的位置。
        實測封包摸 8 萬、手上有西,畫面的摸牌位卻是西。照封包算的話西在第 13 格,而那裡實際
        是 8 索,於是「要打西,打出去是 8 索」。莊家第一手 7 次錯 5 次,其他 254 手全對。
        """
        order = sorted(self._hand.tiles, key=sort_key)
        if pai not in order:
            logger.warning("引擎要打 {},但手上沒有 —— 手牌 {}", pai, order)
            return None
        index = order.index(pai)
        last = len(order) - 1
        return tile_point(index, last, drawn=index == last)

    def act(self, page: Page) -> None:
        now = time.monotonic()
        self._heartbeat(now)
        if not self.allowed:
            return
        if self._unconfirmed is not None:
            self._retry(page, now)
            return
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
        if not self._pending and not self.dry_run:
            # **按下去之前就要記。** 按壓的這段時間 Playwright 會送事件進來,而打出去的回應
            # 常常就在這時候到。按完才記的話,那個回應已經過了,成功的一下會被當成沒反應。
            self._unconfirmed = (fx, fy)
            self._retry_at = now + _RETRY_AFTER
            self._retries = 0
        if not self.dry_run:
            _press(page, x, y)
        self._last_click = now
        self.actions += 1
        if not self._pending:
            self._ready_at = None

    def _retry(self, page: Page, now: float) -> None:
        """剛才那一下沒有下文,再點一次。見 :data:`_RETRY_AFTER`。"""
        if now < self._retry_at or self._unconfirmed is None:
            return
        if self._retries >= _MAX_RETRIES:
            logger.warning("點了 {} 次都沒有反應,放棄這一手", self._retries + 1)
            self._unconfirmed = None
            return
        size = viewport(page)
        if size is None:
            return
        fx, fy = self._unconfirmed
        self._retries += 1
        logger.info("沒有反應,第 {} 次重點 ({:.3f}, {:.3f})", self._retries, fx, fy)
        _press(page, int(size[0] * fx), int(size[1] * fy))
        self._retry_at = time.monotonic() + _RETRY_AFTER

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
            "自動打牌:封包 {} / 事件 {} / 動作 {} / 點錯 {} / 手牌 {} 張,摸到 {} / {}",
            self.frames,
            self.events,
            self.actions,
            self.misplays,
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


# --------------------------------------------------------------------- 開房間

#: 下面這些座標都是畫布比例,在 1280x720 的大廳量的。

#: 大廳的「友人場」,以及點下去之後的「創設房間」。
_FRIEND_ROOM = (0.72, 0.70)
_CREATE_ROOM = (0.727, 0.375)
#: 建立房間頁。按鈕正中央是字,判斷顏色要取沒有字的地方,所以另外記。
_FOUR_PLAYERS_MARK = (0.225, 0.27)
_HANCHAN = (0.554, 0.468)
_HANCHAN_MARK = (0.515, 0.468)
_CREATE = (0.5, 0.875)
_CREATE_MARK = (0.445, 0.875)
#: 房間裡三個空位的「新增電腦」。
_ADD_CPU = ((0.334, 0.565), (0.533, 0.565), (0.73, 0.565))
_START = (0.594, 0.925)
#: 「開始」要三台電腦都就位才會從灰色變橘色。
_START_MARK = (0.54, 0.925)

#: 高級設置。開啟後設定列表要往下捲 :data:`_SCROLL_STEPS` 格才看得到下面兩項。
_ADVANCED = (0.417, 0.663)
_SETTINGS = (0.5, 0.55)
_SCROLL_STEPS = 9
#: 滾輪一格。列表每格的位移是固定的,與這個數字無關,只要夠大。
_SCROLL_WHEEL = 160
#: 兩格之間等列表滑完。
_SCROLL_MS = 400
#: 捲到底之後的「1 位必要點數」。按下去是遊戲自己的數字鍵盤,不吃實體鍵盤。
_FIRST_PLACE = (0.278, 0.495)
_PAD_TWO = (0.452, 0.434)
_PAD_ZERO = (0.452, 0.681)
_PAD_OK = (0.4535, 0.794)
#: 捲到底之後「擊飛」的「關閉」。
_NO_BUST = (0.278, 0.595)
#: 延長局的 1 位必要點數。最大值,沒有人到得了,半莊一定打進西入。
_FIRST_PLACE_POINTS = 200000

#: 大廳的公告欄右上角紅底白字的「×」。新活動開跑那天登入會先跳公告欄,蓋住整個大廳,
#: 2026-11-09 就是在這上面點了「友人場」。量自 1280x720 的截圖:中心是白色的叉,上下左右是紅底。
_NOTICE_CLOSE = (0.8475, 0.175)
_NOTICE_RED = ((0.8475, 0.16), (0.8475, 0.19), (0.835, 0.175), (0.86, 0.175))
#: 實測紅底 (84, 72, 180) 與 (75, 56, 168),白叉 (194, 210, 241)。
_RED_LOW, _RED_HIGH = (50, 35, 140), (115, 105, 220)
_WHITE_LOW = (170, 180, 215)
#: 公告欄後面接著跳的活動簽到(2026-11-09 聯動紀念簽到),沒有「×」,只能按「簽到」。
#: 認法:「簽到」鈕左右兩側橘色,彈窗左邊米白底。實測米白 (232, 246, 247)、(242, 255, 255)。
_SIGN_IN = (0.5, 0.87)
_SIGN_IN_ORANGE = ((0.43, 0.87), (0.57, 0.87))
_SIGN_IN_PAPER = ((0.3, 0.53), (0.25, 0.3))
_PAPER_LOW = (215, 230, 230)
#: 簽完之後同一個位置變成淺藍的「確認」,按了才關。實測 (212, 183, 136)、(216, 189, 136)。
_CONFIRM_LOW, _CONFIRM_HIGH = (180, 150, 105), (240, 215, 170)
#: 大廳的彈窗關了還在就再關,最多這麼多次,之後照常往下走、由各步的顏色檢查擋住。
_POPUP_TRIES = 4

#: 橘色按鈕,BGR 的範圍。實測選中的選項是 (70, 175, 255),「創建」是 (58, 154, 239),
#: 沒選中的是藍灰色 (104, 77, 51),大廳同一個位置是 (54, 31, 142) 與 (203, 204, 221)。
_ORANGE_LOW = (30, 130, 210)
_ORANGE_HIGH = (110, 210, 255)

#: 登入之後等多久才開始點。大廳要先演進場動畫。
_LOBBY_SETTLE = 6.0
#: 兩次點擊之間等多久。頁面切換有動畫,太快會點在舊頁面上。
_STEP_DELAY = 2.0
#: 點了之後等多久要看到封包確認。
_REPLY_TIMEOUT = 15.0
#: 可以重按的那一步,多久沒回覆就再按一次。實測第三台電腦的點擊沒有送出請求,房間就開不成。
_RETRY_AFTER = 4.0

_LOGIN = ".lq.Lobby.loginSuccess"
_CREATE_REQUEST = ".lq.Lobby.createRoom"
_ROOM_UPDATE = ".lq.NotifyRoomPlayerUpdate"

_HANCHAN_MODE = 2
_FOUR_PLAYERS = 4


def _within(
    image: np.ndarray, fx: float, fy: float, low: tuple[int, ...], high: tuple[int, ...]
) -> bool:
    """畫面上那一點的顏色在不在範圍裡。取一小塊平均,單一像素可能剛好在花紋上。"""
    height, width = image.shape[:2]
    x, y = int(width * fx), int(height * fy)
    patch = image[max(0, y - 4) : y + 4, max(0, x - 4) : x + 4].reshape(-1, 3)
    if patch.size == 0:
        return False
    mean = patch.mean(axis=0)
    return all(lo <= value <= hi for value, lo, hi in zip(mean, low, high, strict=True))


def is_orange(image: np.ndarray, fx: float, fy: float) -> bool:
    """畫面上那一點是不是橘色按鈕。"""
    return _within(image, fx, fy, _ORANGE_LOW, _ORANGE_HIGH)


def notice_open(image: np.ndarray) -> bool:
    """大廳的公告欄開著:「×」中心白、四周紅。"""
    return _within(image, *_NOTICE_CLOSE, _WHITE_LOW, (255, 255, 255)) and all(
        _within(image, fx, fy, _RED_LOW, _RED_HIGH) for fx, fy in _NOTICE_RED
    )


def sign_in_open(image: np.ndarray) -> bool:
    """大廳蓋著活動簽到:米白底、下方橘色的「簽到」。"""
    return all(is_orange(image, fx, fy) for fx, fy in _SIGN_IN_ORANGE) and all(
        _within(image, fx, fy, _PAPER_LOW, (255, 255, 255)) for fx, fy in _SIGN_IN_PAPER
    )


def signed_in_open(image: np.ndarray) -> bool:
    """活動簽到已經簽完,等著按「確認」。"""
    return all(
        _within(image, fx, fy, _CONFIRM_LOW, _CONFIRM_HIGH) for fx, fy in _SIGN_IN_ORANGE
    ) and all(_within(image, fx, fy, _PAPER_LOW, (255, 255, 255)) for fx, fy in _SIGN_IN_PAPER)


#: 大廳可能蓋著的彈窗:(名稱, 認得出來嗎, 按哪裡關掉)。
_POPUPS: tuple[tuple[str, Callable[[np.ndarray], bool], tuple[float, float]], ...] = (
    ("公告欄", notice_open, _NOTICE_CLOSE),
    ("活動簽到", sign_in_open, _SIGN_IN),
    ("簽到完的確認", signed_in_open, _SIGN_IN),
)


@dataclass(frozen=True)
class _Stage:
    """開房間的一步。"""

    name: str
    point: tuple[float, float]
    #: 按之前這些位置要是橘色,否則就不在該在的頁面上。
    marks: tuple[tuple[float, float], ...] = ()
    #: 按了之後等這件事成立才往下走。``None`` 就只等動畫。
    done: Callable[[CpuRoomPilot], bool] | None = None
    #: 不是按,是捲動設定列表這麼多格,負的往上。
    scroll: int = 0
    #: 沒回覆可以在同一點再按。只給「新增電腦」:同一個空位,上一下其實生效的話那裡已經是電腦的
    #: 卡片,再按也沒事。其他步驟按錯會到別的頁面,不重按。
    retry: bool = False


def _overtime_stages() -> tuple[_Stage, ...]:
    """打開高級設置,1 位必要點數設成最大、關掉擊飛:一定打得進西入。

    擊飛要關:Mortal 打簡單的電腦,實測五場有兩場不到南場就有人被飛掉。
    """
    digits = (_PAD_TWO, *(_PAD_ZERO,) * 5)
    return (
        _Stage("高級設置", _ADVANCED),
        _Stage("往下捲", _SETTINGS, scroll=_SCROLL_STEPS),
        _Stage("1 位必要點數", _FIRST_PLACE),
        *(_Stage("數字", point) for point in digits),
        _Stage("確認", _PAD_OK),
        _Stage("關閉擊飛", _NO_BUST),
        _Stage("捲回去", _SETTINGS, scroll=-_SCROLL_STEPS),
    )


def _stages(*, overtime: bool = False) -> tuple[_Stage, ...]:
    def robots(count: int) -> Callable[[CpuRoomPilot], bool]:
        return lambda pilot: pilot.robots >= count

    return (
        _Stage("友人場", _FRIEND_ROOM),
        _Stage("創設房間", _CREATE_ROOM),
        _Stage("半莊", _HANCHAN, marks=(_FOUR_PLAYERS_MARK, _CREATE_MARK)),
        *(_overtime_stages() if overtime else ()),
        _Stage("創建", _CREATE, marks=(_HANCHAN_MARK,), done=lambda pilot: pilot.room),
        *(
            _Stage("新增電腦", point, done=robots(index + 1), retry=True)
            for index, point in enumerate(_ADD_CPU)
        ),
        _Stage("開始", _START, marks=(_START_MARK,), done=lambda pilot: pilot.started),
    )


@dataclass
class CpuRoomPilot:
    """從大廳開一間四人半莊,加三台電腦,按開始,然後把畫面交給 ``bot``。

    滿足 :class:`~mia.groundtruth.cdp.AutoPlayer`。

    **大廳點錯的後果比牌桌上大。** 同一個位置在另一頁是「段位場」,一路盲點下去有機會排進
    真人對局,而那時候沒有人在電腦前面。所以每一步都要有憑據才往下走:進建立房間頁之後看
    按鈕顏色,開房、加電腦、開始都等封包回覆。任何一步對不上就存畫面、設 :attr:`failed`,
    不重試也不猜。
    """

    bot: Any
    #: 失敗時的畫面存在這裡。
    out: Path
    schema: LiqiSchema | None = None
    #: 開成一定打進西入的房間,見 :func:`_overtime_stages`。
    overtime: bool = False
    #: 為什麼停下來。空字串表示沒事。
    failed: str = ""
    #: 房間開好了,而且是四人半莊。
    room: bool = False
    #: 房間裡有幾台電腦。
    robots: int = 0
    #: 進牌桌了。
    started: bool = False

    _parsers: dict[str, LiqiParser] = field(default_factory=dict, repr=False)
    _plan: tuple[_Stage, ...] = field(default_factory=_stages, repr=False)
    _index: int = field(default=0, repr=False)
    _wait_until: float | None = field(default=None, repr=False)
    _deadline: float | None = field(default=None, repr=False)
    _retry_at: float | None = field(default=None, repr=False)
    _popup_tries: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.schema is None:
            self.schema = LiqiSchema.load()
        self._plan = _stages(overtime=self.overtime)

    # ------------------------------------------------------------ 封包

    def observe(self, payload: bytes, *, from_client: bool, flow: str) -> None:
        self.bot.observe(payload, from_client=from_client, flow=flow)
        if self.started:
            return
        parser = self._parsers.setdefault(flow, LiqiParser(self.schema))
        try:
            message = parser.parse(payload, from_client=from_client)
        except LiqiParseError:
            return
        self.note(message)

    def note(self, message: LiqiMessage) -> None:
        """看一則訊息,更新憑據。"""
        data: Any = message.payload
        if message.method == _LOGIN and message.kind is MessageKind.RESPONSE:
            if self._wait_until is None:
                self._wait_until = time.monotonic() + _LOBBY_SETTLE
        elif message.method == _CREATE_REQUEST and message.kind is MessageKind.RESPONSE:
            mode, count = data.room.mode.mode, data.room.max_player_count
            if mode != _HANCHAN_MODE or count != _FOUR_PLAYERS:
                self._fail(f"開出來的房間不是四人半莊(mode={mode},人數={count})")
            elif not self.overtime:
                self.room = True
            else:
                rule = data.room.mode.detail_rule
                if rule.fandian == _FIRST_PLACE_POINTS and not rule.can_jifei:
                    self.room = True
                else:
                    self._fail(f"延長局沒設好(必要點數 {rule.fandian},擊飛 {rule.can_jifei})")
        elif message.method == _ROOM_UPDATE:
            if len(data.player_list) > 1:
                self._fail("房間裡出現了其他真人")
            else:
                self.robots = len(data.robots)
        elif message.method == _AUTH_GAME and message.kind is MessageKind.RESPONSE:
            self.started = True

    def _fail(self, reason: str) -> None:
        if not self.failed:
            logger.error("開房間失敗:{}", reason)
            self.failed = reason

    # ------------------------------------------------------------ 畫面

    def act(self, page: Page) -> None:
        if self.started:
            self.bot.act(page)
            return
        if self.failed or self._wait_until is None:
            return
        now = time.monotonic()
        if now < self._wait_until:
            return
        if self._deadline is not None:
            last = self._plan[self._index - 1]
            if last.done is not None and not last.done(self):
                if now > self._deadline:
                    self._give_up(page, f"按了「{last.name}」等不到回覆")
                elif last.retry and self._retry_at is not None and now > self._retry_at:
                    self._press_again(page, last)
                return
            self._deadline = None
        if self._index >= len(self._plan):
            return
        stage = self._plan[self._index]
        if self._index == 0 and self._close_popup(page):
            return
        if stage.marks:
            image = _grab(page)
            if image is None or not all(is_orange(image, *mark) for mark in stage.marks):
                self._give_up(page, f"要按「{stage.name}」時畫面不對")
                return
        size = viewport(page)
        if size is None:
            return
        logger.info("開房間:按「{}」", stage.name)
        x, y = int(size[0] * stage.point[0]), int(size[1] * stage.point[1])
        if stage.scroll:
            page.mouse.move(x, y)
            for _ in range(abs(stage.scroll)):
                page.mouse.wheel(0, _SCROLL_WHEEL if stage.scroll > 0 else -_SCROLL_WHEEL)
                page.wait_for_timeout(_SCROLL_MS)
        else:
            _press(page, x, y)
        self._index += 1
        self._wait_until = time.monotonic() + _STEP_DELAY
        if stage.done is not None:
            self._deadline = time.monotonic() + _REPLY_TIMEOUT
            self._retry_at = time.monotonic() + _RETRY_AFTER

    def _close_popup(self, page: Page) -> bool:
        """大廳蓋著認得的彈窗就先關掉。關了回 True,等動畫完再看一次。"""
        if self._popup_tries >= _POPUP_TRIES:
            return False
        image = _grab(page)
        size = viewport(page)
        if image is None or size is None:
            return False
        for name, seen, (fx, fy) in _POPUPS:
            if seen(image):
                self._popup_tries += 1
                logger.info("開房間:大廳蓋著{},先關掉", name)
                _press(page, int(size[0] * fx), int(size[1] * fy))
                self._wait_until = time.monotonic() + _STEP_DELAY
                return True
        return False

    def _press_again(self, page: Page, stage: _Stage) -> None:
        size = viewport(page)
        if size is None:
            return
        logger.info("開房間:「{}」沒有回覆,再按一次", stage.name)
        _press(page, int(size[0] * stage.point[0]), int(size[1] * stage.point[1]))
        self._retry_at = time.monotonic() + _RETRY_AFTER

    def _give_up(self, page: Page, reason: str) -> None:
        self._fail(reason)
        try:
            self.out.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(self.out / "lobby-failed.png"))
        except Exception as exc:  # noqa: BLE001 - 存不下來不值得蓋掉原本的錯誤
            logger.warning("存不下畫面:{}", exc)


def _grab(page: Page) -> np.ndarray | None:
    """現在的畫面,BGR。"""
    try:
        data = page.screenshot()
    except Exception as exc:  # noqa: BLE001 - 分頁可能剛好在換頁
        logger.warning("截不到畫面:{}", exc)
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
