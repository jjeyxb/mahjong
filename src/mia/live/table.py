"""畫面 → 整桌狀態 → 放銃危險度。**完全不靠封包。**

與 :class:`~mia.live.danger.DangerWorker` 是同一件事的兩條路:那邊讀封包,
這邊只看畫面。兩邊最後都產生一個 :class:`~mia.mjai.table.TableTracker`,
再交給同一支 :func:`~mia.analysis.danger.assess` —— 這正是「用 CV 完成 MJAI
事件流」的兌現點。

為什麼是獨立的一條慢路
----------------------
手牌與牌河的**節奏差一個數量級**:

* 手牌每一巡都變,而且使用者正盯著它做決定 —— 要跟手。
* 牌河每家大約**五秒**才多一張。

成本也差一個數量級(2536 牌桌實測):

==========================  ==========
判有無(四家 18 格)         3.5 ms
認一格                      19.6 ms
認全部 65 格                1266 ms
``find_sideways`` 四家      1830 ms
寶牌面板                    10 ms
==========================  ==========

全套三秒,15 fps 的迴圈完全吃不下。但**牌河不需要 15 fps**。所以這裡是一條獨立的
執行緒,擷取那條路把最新的畫面丟進來,這邊照自己的節奏吃:一秒走完一輪,中間只花幾毫秒
看河裡有沒有新牌,有牌停穩了、或手牌變了,就立刻走一輪。見 :meth:`TableWorker._watch_landing`。

省錢的辦法是「只認新的」
------------------------
每次重認全部是 1266 毫秒,而牌河每次只多一張。所以:

* 張數沒變 → **什麼都不用認**
* 多了 k 張 → 只認那 k 格
* 少一張(被鳴走)→ 整家重認(罕見,一場約九次)

「張數沒變就不重認」能成立,是因為量過:**牌河的未知不會被時間解決**。
牌不會動,同一塊像素讀一百次還是同一個答案(觀察次數翻四倍,補上的未知
一個都沒多)。所以重讀未知格是純粹的浪費。見 ``docs/cv_mjai.md`` §1.6。
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace

import cv2
import numpy as np

from mia.analysis.danger import assess
from mia.calibration.table import Calibration
from mia.config.models import RoiConfig
from mia.live.bus import Dangers, UpdateBus, WorkerStatus
from mia.mjai.events import Dahai, Dora, Reach, Tsumo
from mia.mjai.table import TableTracker
from mia.mjai.tiles import ms_to_mjai, normalize_red
from mia.tracker import (
    UNKNOWN,
    DoraWatcher,
    HandWatcher,
    KyokuLog,
    KyokuSignal,
    KyokuWatcher,
    MeldWatcher,
    Move,
    OwnMove,
    ReachWatcher,
    RiverWatcher,
    RoundWatcher,
    Step,
    Stream,
    TurnTracker,
    compose,
    guess_meld,
    resynced,
)
from mia.tracker.melds import KAN_WIDTH
from mia.utils.geometry import Rect
from mia.utils.logging import logger
from mia.vision.hud import Hud, HudDigits, read_hud, read_wall
from mia.vision.roi import MissingRoiError, RoiSet
from mia.vision.tiles.classify import (
    DEFAULT_SKIN,
    MIN_BRIGHTNESS,
    WHOLE_RED,
    Match,
    TemplateSet,
    classify,
    red_ink,
    settle_five,
)
from mia.vision.tiles.dora import read_dora
from mia.vision.tiles.melds import DEFAULT_VIEWS as MELD_VIEWS
from mia.vision.tiles.melds import (
    GROUP_LENGTH,
    MeldFaces,
    occupied_span,
    read_meld_faces,
)
from mia.vision.tiles.river import (
    COLUMNS,
    CONFIDENT,
    DEFAULT_CORNERS,
    SEATS,
    River,
    classify_river_tile,
    classify_sideways,
    face_patch,
    face_size,
    find_sideways,
    read_river,
    search_window,
)

__all__ = ["TableLink", "TableWorker"]

#: 兩次讀取之間至少隔多久。牌河每家約五秒才多一張 —— 一秒一次綽綽有餘,
#: 而且讓出 CPU 給擷取那條路(它要維持 15 fps)。
_INTERVAL = 1.0

#: 多久看一次河裡有沒有新牌。只攤平、判有無,實測一次 6 毫秒,比整輪便宜得多。
#:
#: 一秒看一次的話別家的牌平均晚半秒才進事件流,而那正是要問吃碰的時候。不能再短:看的是兩次之間
#: 牌有沒有動,太短時自己打出去的牌飛得慢,兩次看起來一樣。實測 0.1 秒自己的牌有 36% 讀在飛行中,
#: 0.15 秒 5%。
_POLL = 0.15

#: 新牌的牌面兩次之間差多少以內算停穩了,每個通道的平均差。
_STILL = 3.0

#: 副露長度至少隔這麼久量一次,單位是輪。:class:`~mia.tracker.melds.MeldWatcher` 要連讀兩次一樣才
#: 採信,而牌桌現在有牌落下就走一輪:隔太近的話兩次都讀到鳴牌動畫的過渡值,實測多出一組假的碰。
_MELD_GAP = 0.4

#: 上家的座位,依視角。
_KAMICHA = 3

#: 吃碰時手上拿出去幾張。
_OPEN_CALL = 2

#: 旋轉模板的分數要比直放高多少,才當成橫放的牌。見 :meth:`TableWorker._turned`。
_TURNED = 0.03

#: 寶牌面板在牌桌矩形裡的位置(``data/roi_ref/roi.yaml`` 的 ``dora_indicators``)。
_DORA_ROI = "dora_indicators"


#: 四家的牌河至少幾家讀得到,這一輪的副露長度才採信。見 ``_read_melds``。
_MELD_QUORUM = 2

#: 白(雀魂記法 ``5z`` 經 :func:`ms_to_mjai` 之後)。見 ``_read_seat``。
_HAKU = "P"

#: 一張捨牌:誰打的、什麼牌(讀不出來是 ``None``)、猜是什麼牌、是他的第幾張。
_Seen = tuple[int, str | None, str | None, int]

#: 開局余牌:配牌之後是 70,莊家摸完 69。見 ``_read_round``。
_OPENING_WALL = (70, 69)

#: 河裡多一張之後,等手牌跟上幾輪。見 ``_catch_up_own``。
_OWN_PATIENCE = 2

#: 自己的河最多比手牌交代的多幾張。手牌會慢一兩手,再多就不是牌了。見 ``_own_river_covered``。
_OWN_RIVER_LEAD = 2

#: 河裡那張手上也有時,等幾輪。多半是從手牌打出去的,新的手牌讀數只是還沒到,先補的話
#: 讀數一到,同一次摸打就記了兩次(實測一局 13 手有 9 手因此晚一輪)。但手上有同一張
#: 還摸切也是有的,所以是等久一點,不是不補。
_OWN_PATIENCE_HELD = 4

#: 連續幾副不同的手牌解釋不了,就算事件流對不上了。見 ``stream``。
_HAND_LOST = 3

#: 事件流停住這麼多秒就用畫面重寫自己的動作。手牌慢一步是常有的事,實測最久 3 秒多一點,等一下
#: 就接上了;重寫太早只是白白失去摸切的標記。見 :func:`~mia.tracker.stream.resynced`。
_STALL_PATIENCE = 3.0

#: 組長量的時候,第一組開頭有多少空白,單位是格。比這個多的部分不算牌寬。
_MELD_LEAD = 0.3

#: 排出來的摸牌數與余牌差到幾張,就算事件流對不上了。排序本來就會落後幾步。
_WALL_LOST = 5

#: 一輪湧進這麼多張捨牌,代表是中途接上,或畫面被蓋了好幾秒:一張一張排已經排不出來,改照輪替
#: 整批補上,見 :meth:`~mia.tracker.turns.TurnTracker.jump`。CPU 對局一秒最多兩個動作,實測一輪
#: 最多 3 張。
_JOINED_LATE = 4


def _sure(match: Match) -> bool:
    """牌河這一格的答案有把握。"""
    return (
        match.score >= CONFIDENT[0]
        and match.margin >= CONFIDENT[1]
        and match.brightness >= MIN_BRIGHTNESS
    )


@dataclass
class _OwnTold:
    """自己的動作手牌與畫面各會看到一次,排序器只該聽一次。"""

    told: int = 0
    shown: int = 0

    def seen(self) -> bool:
        """畫面看到一次,回傳排序器還不知道。"""
        self.shown += 1
        fresh = self.shown > self.told
        self.told = max(self.told, self.shown)
        return fresh

    def tell(self, count: int) -> range:
        """手牌說總共發生過 ``count`` 次,回傳還沒告訴排序器的那幾次。"""
        fresh = range(self.told, count)
        self.told = max(self.told, count)
        return fresh


@dataclass(frozen=True, slots=True)
class _PendingFaces:
    """一組還在認牌面的副露:認好之後填到 ``table`` 的那一家第 ``index`` 組。

    ``table`` 是為了換局:``_new_kyoku`` 會換掉 ``TableWorker._table``,
    對不上就丟掉。
    """

    table: TableTracker
    position: int
    index: int
    future: Future[MeldFaces]


class TableWorker(threading.Thread):
    """只看畫面,追蹤整桌狀態,把每張牌的危險度投進郵箱。

    Args:
        bus: 結果往哪投。
        config: ROI 設定,用來切寶牌面板。
        skin: 牌面皮膚。
        interval: 兩次讀取之間至少隔多久,秒。
        link: 轉接點,收尾時會從它身上拔掉自己。
        templates: 先載好的模板集。``None`` 就在執行緒裡自己載 ——
            載模板要讀 37 個檔,不該卡住造物件的那一方。
        on_stream: 每一輪結束時收 ``(第幾局, 事件流)``。要事件流的人才給:
            沒有人要的話不必組。

    Note:
        **座位是依視角的**,自己永遠是 0。Mortal 看的是相對座位,所以事件流也這樣
        送(``start_game`` 的 ``id`` 是 0),不必換算成絕對座位。
    """

    #: 給人看的「這份危險度從哪來」。見 :data:`~mia.live.danger.DangerWorker.SOURCE`。
    SOURCE = "圖像辨識"

    def __init__(
        self,
        bus: UpdateBus,
        *,
        config: RoiConfig,
        skin: str = DEFAULT_SKIN,
        interval: float = _INTERVAL,
        link: TableLink | None = None,
        templates: TemplateSet | None = None,
        on_stream: Callable[[int, Stream], None] | None = None,
    ) -> None:
        super().__init__(name="table", daemon=True)
        self._on_stream = on_stream
        self._link = link
        self._bus = bus
        self._config = config
        self._rois: RoiSet | None = None
        self._panel: Rect | None = None
        self._skin = skin
        self._interval = interval
        self._lock = threading.Lock()
        self._frame: tuple[np.ndarray, Calibration] | None = None
        self._hand: tuple[str, ...] = ()
        # 兩輪之間手牌可能變了好幾次(摸、打),一次都不能漏,所以是佇列。
        self._hands: list[tuple[tuple[str, ...], str | None]] = []
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._began: float | None = None
        self._last_round = 0.0
        self._templates = templates
        self._river_templates = templates
        self._reds: TemplateSet | None = None
        self._digits: HudDigits | None = None

        self._table = TableTracker()
        # 視角位置直接當絕對座位,自己是 0 —— 於是「下家」就是 1,與
        # SEATS 的順序一致,UI 的相對標籤不必再換算一次。
        self._table.seat = 0
        self._rivers = RiverWatcher()
        self._dora = DoraWatcher()
        self._kyoku = KyokuWatcher()
        self._round = RoundWatcher()
        self._reach = ReachWatcher()
        self._turns: TurnTracker | None = None
        self._turn_buffer: list[tuple[float, list[_Seen], list[int], list[int], int | None]] = []
        self._turn_discards: list[_Seen] = []
        self._turn_melds: list[int] = []
        self._turn_kans: list[int] = []
        self.steps: list[Step] = []
        self._own = HandWatcher()
        self._own_behind = 0
        self._own_drops = _OwnTold()
        self._own_calls = _OwnTold()
        self._own_counted = 0
        self._kamicha_then = 0
        self._meld_read = -_MELD_GAP
        self._dora_marks: list[tuple[int, str]] = []
        self._reach_marks: dict[int, int] = {}
        self._own_reach: int | None = None
        self._readings: dict[tuple[int, int], tuple[str | None, str | None]] = {}
        self._melds = MeldWatcher(units=tuple(GROUP_LENGTH[seat] for seat in SEATS))
        # 認牌面慢(一組約 0.5 秒),放到另一條執行緒。見 _read_melds。
        self._faces = ThreadPoolExecutor(max_workers=1, thread_name_prefix="meld-faces")
        self._pending_faces: list[_PendingFaces] = []
        self._meld_reds: dict[tuple[int, int], frozenset[str]] = {}
        self._known: list[list[str]] = [[] for _ in SEATS]
        # 每家每一格的最佳猜測,沒信心的也留著:Mortal 不收 ?。
        self._guess: list[dict[int, str]] = [{} for _ in SEATS]
        self._sideways: list[int | None] = [None for _ in SEATS]
        # 每家「上一輪讀成白、還沒收下」的那一格。見 _read_seat。
        self._haku: list[int | None] = [None for _ in SEATS]
        # 每家還沒讀定的格子上一次長什麼樣、這一次哪幾格停穩了。見 _watch_landing。
        self._landing: list[dict[int, np.ndarray]] = [{} for _ in SEATS]
        self._still: list[set[int]] = [set() for _ in SEATS]
        self._clear: list[bool] = [False for _ in SEATS]
        # 轉過來比較像的格子,見 _turned。
        self._leaning: list[set[int]] = [set() for _ in SEATS]
        # 動畫中途先讀的格子,停穩之後要重讀。
        self._unsteady: list[set[int]] = [set() for _ in SEATS]
        self._cells: dict[tuple[int, int, int], tuple[bytes, str, str]] = {}
        self._dora_seen = b""
        self._seen_hand: tuple[list[str], str | None] | None = None
        self._seen_at: float | None = None
        self._stalled_at: float | None = None
        self._last: tuple | None = None
        self._wall: int | None = None
        #: 第幾局,每清空一次桌面加一。事件流靠它知道換局了。
        self.kyoku = 0

        self.status = WorkerStatus("牌桌")
        self.reads = 0
        self.posts = 0

    # ------------------------------------------------------------ 給擷取那條路

    def offer_frame(self, image: np.ndarray, calibration: Calibration) -> None:
        """收一張畫面。**只留最新的** —— 舊的已經沒有意義了。

        與 :class:`~mia.live.bus.UpdateBus` 同一個理由:積起來之後補播,
        螢幕上顯示的會是已經不成立的局面。
        """
        with self._lock:
            self._frame = (image, calibration)
            waiting = bool(self._hands)
        if waiting:
            # 手牌先到、畫面已經被上一輪拿走了:不等滿一秒,這張畫面一到就讀。
            self._wake.set()

    def offer_hand(self, tiles: Sequence[str], drawn: str | None = None) -> None:
        """收一副手牌(雀魂記法,含 ``drawn``)。

        危險度要知道自己手上有什麼才算得出來,事件流要靠它知道自己摸打了什麼。
        """
        with self._lock:
            self._hand = tuple(tiles)
            self._hands.append((tuple(tiles), drawn))
        # 手牌一變就要讀,不等下一輪:輪到自己的那一刻正是 Mortal 要回答的時候。
        # 牌桌一秒讀一次的話,建議平均晚半秒、最多晚一秒,而手牌一手才變兩次。
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    # ------------------------------------------------------------------ 主迴圈

    def run(self) -> None:
        self.status.alive = True
        self._began = time.monotonic()
        if self._templates is None:
            try:
                self._templates = TemplateSet.load(self._skin)
                self._river_templates = TemplateSet.load_river(self._skin)
            except (FileNotFoundError, ValueError) as exc:
                self.status.say(f"載不到牌面模板:{exc}")
                self.status.alive = False
                return
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                try:
                    self.tick()
                except Exception as exc:  # noqa: BLE001 - 執行緒裡漏出來的例外只會被吞掉
                    logger.warning("牌桌辨識這一輪失敗:%s", exc)
                    self.status.say(f"這一輪讀不成:{exc}")
                poll = min(self._interval, _POLL)
                self._wake.wait(max(0.0, poll - (time.monotonic() - started)))
                self._wake.clear()
        finally:
            self.status.alive = False
            self._faces.shutdown(wait=False, cancel_futures=True)
            if self._link is not None:
                self._link.detach(self)

    def tick(self) -> None:
        """讀一次。平常由 :meth:`run` 自己叫,測試可以直接叫。"""
        with self._lock:
            frame = self._frame
            hand = self._hand
            hands = self._hands
            self._hands = []
            self._frame = None
        if frame is None:
            # **不要安靜地 return。** 使用者回報過:只打開放銃分析的話畫面
            # 一直停在「等待對局…」,而這裡拿不到畫面就直接回去,沒有任何
            # 訊息 —— 看起來就是撥了開關什麼都沒發生。
            #
            # 現在 LiveRuntime 會一併打開畫面辨識(見 _ensure_vision_for),
            # 所以這句話只會在真的還沒送到第一幀時出現(找視窗、校正中)。
            #
            # 讀過之後就不講了:手牌把這一輪提早叫醒時,新的畫面可能還沒到。
            if not self.reads:
                self.status.say("等畫面辨識把牌桌送過來")
            with self._lock:
                self._hands = hands + self._hands
            return

        image, calibration = frame
        table = calibration.table_rect
        flats = [DEFAULT_CORNERS[seat].rectify(image, table) for seat in SEATS]
        rivers = [read_river(flat) for flat in flats]
        if not self._round_due(flats, rivers, bool(hands)):
            if self._turns is not None and self._digits is not None:
                self._turns.see_wall(self._turn_clock(), read_wall(image, table, self._digits))
            return
        self.reads += 1
        if not self._load_digits():
            return
        if self._rois is None or not self._rois.matches(calibration):
            self._rois = RoiSet(self._config, calibration)
        self._panel = calibration.panel

        signal = self._kyoku.observe(rivers, taken=self._same_round(image, table, rivers))
        if signal is not KyokuSignal.NONE:
            # 開局訊號要等第一張牌,而自己是莊家時那已經太晚了:配牌、寶牌、HUD
            # 在那之前就收好了,要留著。清空的時候已經重設過一次。
            if signal is KyokuSignal.ENDED or self._round.settled is None:
                self._new_kyoku(keep_hand=signal is KyokuSignal.STARTED)
            self.status.say(
                "新的一局" if signal is KyokuSignal.STARTED else "牌桌清空了"
            )


        # **局還沒開始就不讀牌河。** 開始要四家都連續,而讀牌只看那一家:第一張牌
        # 剛好遇上別家不連續的那一輪,牌會先被收下,下一輪開局訊號到了又整個重設,
        # 同一批牌再發一次 —— 事件流的順序就亂了。開局前的介面也擋在這裡。
        if not self._kyoku.empty:
            for position, (flat, river) in enumerate(zip(flats, rivers, strict=True)):
                if not river.is_plausible():
                    continue
                self._read_seat(position, flat, river, urgent=bool(hands))

        self._read_hands(hands, image, table)
        self._catch_up_own()
        self._read_melds(image, table, rivers)
        self._read_dora(image, table)
        self._read_round(image, table, rivers)
        self._mark_declared()
        self._read_turns(image, table)
        # **狀態寫在最後。** 寫在開頭的話會被這一輪後面的訊息蓋掉 ——
        # 實測「等讀到自己的手牌」被局邊界的「新的一局」洗掉了,而那正是
        # 使用者最需要看到前者的時候。狀態列要講的是這一輪**結束時**的狀態。
        #
        # 危險度是「我切這張會不會放銃」—— 沒有手牌就沒有問題可以回答。
        # 牌河照常讀下去,等手牌來了才算得出來。
        self.status.say("" if hand else "等讀到自己的手牌")
        self._publish(hand)
        if self._on_stream is not None:
            self._on_stream(self.kyoku, self.stream())

    # -------------------------------------------------------------------- 一家

    def _round_due(
        self, flats: Sequence[np.ndarray], rivers: Sequence[River], urgent: bool
    ) -> bool:
        """這次要不要整輪走完。手牌變了、有牌停穩了,或離上一輪夠久了才要。"""
        landed = self._watch_landing(flats, rivers)
        now = time.monotonic()
        if self._began is not None and not (
            urgent or landed or now - self._last_round >= self._interval
        ):
            return False
        self._last_round = now
        return True

    def _watch_landing(self, flats: Sequence[np.ndarray], rivers: Sequence[River]) -> bool:
        """還沒讀定的格子有沒有停穩。

        牌河每格只認一次,所以**什麼時候認**決定了對不對。牌打出來到落定要半秒以上,途中牌面是
        糊的,封包之後 0.3 秒讀的話對不到兩成,而且常常是有信心的白。之前一秒讀一次,剛好碰上動畫
        就錯到局尾,別家捨牌只有 96%。靜止之後同一批牌 99.7%。

        所以等到牌面連續兩次看起來一樣才認,這也比死等一秒快:別家打牌進事件流從中位 1.1 秒變成
        0.6 秒。

        別家的牌可以更早:途中會讀錯的幾乎都是白,所以一看就有把握、又不是白的牌先收,0.48 秒,
        停穩之後再讀一次(:meth:`_revisit`)。1142 張只錯 2 張,與等停穩一樣。自己的牌飛得慢,
        而且牌面手牌已經知道,不必搶這一點時間。
        """
        landed = False
        for position, (flat, river) in enumerate(zip(flats, rivers, strict=True)):
            cells = self._unread(position, river)
            patches = {cell: face_patch(flat, river.cells[cell]) for cell in cells}
            before = self._landing[position]
            still = {
                cell
                for cell, patch in patches.items()
                if cell in before and float(cv2.absdiff(patch, before[cell]).mean()) < _STILL
            }
            self._landing[position] = patches
            self._still[position] = still
            self._clear[position] = self._clear_at_once(position, flat, river, cells, still)
            landed = landed or bool(still) or self._clear[position]
        return landed

    def _clear_at_once(
        self, position: int, flat: np.ndarray, river: River, cells: Sequence[int], still: set[int]
    ) -> bool:
        """這一家下一張新牌還沒停穩,但已經認得出來,而且不是白。"""
        known = len(self._known[position])
        if position == 0 or self._began is None or known not in cells or known in still:
            return False
        if known == self._sideways[position] or known in river.clipped:
            return False
        pai, _ = self._read_cell(position, flat, self._shifted(position, flat, river), known)
        return pai not in (UNKNOWN, _HAKU)

    def _unread(self, position: int, river: River) -> list[int]:
        """這一家還要等它停穩的格子。"""
        if not river.is_plausible() or (position == 0 and self._own_river_covered(river.count)):
            return []
        known = len(self._known[position])
        return [k for k in river.filled if k >= known or k in self._unsteady[position]]

    def _steady(self, position: int, cell: int) -> bool:
        """這一格停穩了。沒有跑 :meth:`run` 時沒有前後兩次可比,就當作停穩了。"""
        return self._began is None or cell in self._still[position]

    def _read_seat(
        self, position: int, flat: np.ndarray, river: River, *, urgent: bool = False
    ) -> None:
        """認一家的河,只認新的那幾格。

        Args:
            urgent: 輪到自己了,還沒停穩的牌也要先讀:Mortal 要知道最新的那張。這種格子停穩之後
                會重讀一次,見 :meth:`_revisit`。
        """
        assert self._templates is not None
        known = self._known[position]
        filled = river.filled
        if position == 0 and self._own_river_covered(len(filled)):
            return
        if len(filled) == len(known):
            self._recheck_reds(position, flat, river, known)
            self._revisit(position, flat, river, known)
            # 沒有新格子,不必重認 —— 但觀察照樣交出去:一口氣長了好幾張的
            # 河要靠**下一次**觀察確認才會發事件(過場動畫的白光,見
            # ``mia.tracker.rivers._JUMP``)。
            self._emit(position, known)
            return
        if len(filled) < len(known):
            # 被鳴走的一定是最後打的那張,前面的牌沒有動。被蓋住的話,讀回來時像素一樣,不必重認。
            del known[len(filled) :]
            self._unsteady[position] = {k for k in self._unsteady[position] if k < len(filled)}

        sideways = self._sideways[position]
        if sideways is None:
            sideways = self._look_for_sideways(position, flat, river, known)
        river = self._shifted(position, flat, river)
        filled = river.filled

        fresh = [k for k in filled if k >= len(known)] if len(known) else list(filled)
        # 整條重認(剛接上、立直之後)不必防白光:那不是白光掃過的時機。
        incremental = bool(known)
        for cell in fresh:
            if cell in river.clipped:
                known.append(UNKNOWN)
                continue
            if cell == sideways:
                # 橫放的讀數不夠當成有把握,所以只當猜測。但事件流的這一張要有答案,
                # 而直放模板對這一格的答案是雜訊:實測 54 張,旋轉 40 對,直放 0 對。
                assert self._river_templates is not None
                view = DEFAULT_CORNERS[SEATS[position]]
                match = classify_sideways(
                    flat, cell, self._river_templates,
                    scale=view.sideways_scale, size=view.sideways_size,
                )
                self._guess[position][cell] = ms_to_mjai(match.label)
                known.append(UNKNOWN)
                continue
            steady = self._steady(position, cell)
            if not steady and not urgent and not self._clear[position]:
                break
            pai, guess = self._read_cell(position, flat, river, cell)
            if not steady and pai in (UNKNOWN, _HAKU) and self._haku[position] != cell:
                # 還在飛的牌只收看得清楚的。先收一個猜測的話,這張牌常常馬上被鳴走,沒有機會停穩重讀:
                # 實測自己碰的那張被記成別的牌,整組副露跟著錯。
                if pai == _HAKU:
                    self._haku[position] = cell
                break
            self._guess[position][cell] = guess
            if incremental and pai == _HAKU and not steady and self._haku[position] != cell:
                # 白要看兩次才收。和了 / 立直的過場有一道白光橫過牌河,空白的
                # 牌面模板對上白光,讀成**有信心的白** —— 而這裡每一格只認一次,
                # 收下就永遠留著,變成一張假現物。重播校正素材:真的一索在白光
                # 掃過的那一刻打出來,被讀成白,上家因此 30 輪都有一張假現物。
                # 這一輪先停在這一格,下一輪重認;真的白只是晚一輪。
                self._haku[position] = cell
                break
            self._haku[position] = None
            known.append(pai)
            if not steady:
                self._unsteady[position].add(cell)

        self._emit(position, known)

    def _shifted(self, position: int, flat: np.ndarray, river: River) -> River:
        """這一家有橫放牌的話,換成讓開它的格線。"""
        sideways = self._sideways[position]
        if sideways is None:
            return river
        scale = DEFAULT_CORNERS[SEATS[position]].sideways_scale
        return read_river(flat, sideways=sideways, scale=scale)

    def _read_cell(
        self, position: int, flat: np.ndarray, river: River, cell: int
    ) -> tuple[str, str]:
        """認一格,回傳 ``(牌, 猜測)``,沒把握時牌是 :data:`UNKNOWN`。

        像素沒變就用上次的答案:河被立繪蓋住、鳴牌或立直重排時要重讀一整家,一家 300 毫秒,實測
        最久一輪 0.5 秒,而那段時間自己的手牌在排隊。
        """
        assert self._templates is not None and self._river_templates is not None
        view = DEFAULT_CORNERS[SEATS[position]]
        rect = river.cells[cell]
        window = search_window(flat, rect, left=view.left_slack)
        key = (position, rect.x, rect.y)
        pixels = hashlib.blake2b(window.tobytes(), digest_size=16).digest()
        memo = self._cells.get(key)
        if memo is not None and memo[0] == pixels:
            return memo[1], memo[2]
        size = face_size(seat=SEATS[position], row=cell // COLUMNS)
        match = classify_river_tile(window, self._templates, size=size)
        # 只有沒把握的才比第二份,86% 的格子用不到:一格 56 → 38 毫秒,讀對的張數一樣,有信心的
        # 還多一點。牌桌每多花一毫秒,輪到自己時的建議就晚一毫秒。
        if not _sure(match) and self._river_templates is not self._templates:
            match = classify_river_tile(window, self._river_templates, size=size)
        label = settle_five(match.label, face_patch(flat, rect))
        guess = ms_to_mjai(label)
        pai = guess if _sure(match) else UNKNOWN
        if pai == UNKNOWN and self._sideways[position] is None:
            guess = self._turned(position, flat, cell, match) or guess
        self._cells[key] = (pixels, pai, guess)
        return pai, guess

    def _turned(self, position: int, flat: np.ndarray, cell: int, upright: Match) -> str | None:
        """沒把握的牌如果轉過來比較像,那就是宣言牌,回傳它是什麼。

        要等點數才知道誰立直,而那一兩秒正是最需要知道現物的時候。實測 39 張宣言牌的猜測 1 → 34 張
        對;沒把握的普通牌 37 張只有 2 張走到這裡,都沒有改錯。只當猜測,不算立直:假立直的代價太大。
        """
        assert self._river_templates is not None
        view = DEFAULT_CORNERS[SEATS[position]]
        turned = classify_sideways(
            flat, cell, self._river_templates, scale=view.sideways_scale, size=view.sideways_size
        )
        if turned.score <= upright.score + _TURNED:
            return None
        self._leaning[position].add(cell)
        return ms_to_mjai(turned.label)

    def _revisit(
        self, position: int, flat: np.ndarray, river: River, known: list[str]
    ) -> None:
        """動畫中途先讀的格子,停穩之後再讀一次,答案不同就改掉。"""
        river = self._shifted(position, flat, river)
        for cell in sorted(self._unsteady[position]):
            if not self._steady(position, cell):
                continue
            self._unsteady[position].discard(cell)
            if cell >= len(known) or cell == self._sideways[position] or cell in river.clipped:
                continue
            pai, guess = self._read_cell(position, flat, river, cell)
            self._guess[position][cell] = guess
            if pai == known[cell]:
                continue
            if known[cell] == UNKNOWN:
                self._table.handle(Dahai(actor=position, pai=pai, tsumogiri=False))
            elif pai != UNKNOWN:
                self._table.revise(position, known[cell], pai)
            known[cell] = pai
            self._rivers.revise(position, cell, pai)
            serials = self._rivers.serials[position]
            if cell < len(serials):
                self._readings[(position, serials[cell])] = (
                    None if pai == UNKNOWN else pai,
                    guess,
                )

    def _own_river_covered(self, seen: int) -> bool:
        """自己的河比手牌交代的多太多,那不是牌。

        吃牌有好幾種組合時,選擇框蓋在自己牌河的第三列,框裡的牌讀起來就是河裡的牌。實測一輪多
        出四張假牌。自己打過幾張手牌最清楚,河最多只比它多 :data:`_OWN_RIVER_LEAD` 張。
        """
        thrown = sum(move.move is Move.DISCARD for move in self._own.moves)
        return bool(self._own.moves) and seen > thrown + _OWN_RIVER_LEAD

    def _recheck_reds(
        self, position: int, flat: np.ndarray, river: River, known: list[str]
    ) -> None:
        """認不出來的格子如果整張紅,只拿赤五與中再認一次。

        每一格只認一次,而剛打出來的牌還在動畫裡:實測一張赤 5 筒在第一幀被猜成白,
        幾幀之後已經認得出來卻不會再認。其他牌沒有這麼便宜的訊號,看顏色只要幾十微秒,
        所以每一輪都看得起。

        其他牌試過整套重認:真人對局與 CPU 半莊都沒有變好,槓的素材反而少了一次鳴牌。
        """
        assert self._templates is not None
        if UNKNOWN not in known:
            return
        if self._reds is None:
            self._reds = self._templates.only("0m", "0p", "0s", "7z")
        sideways = self._sideways[position]
        if sideways is not None:
            scale = DEFAULT_CORNERS[SEATS[position]].sideways_scale
            river = read_river(flat, sideways=sideways, scale=scale)
        for cell, pai in enumerate(known):
            if (
                pai != UNKNOWN
                or cell == sideways
                or cell in river.clipped
                or not river.occupied[cell]
            ):
                continue
            if red_ink(face_patch(flat, river.cells[cell])) < WHOLE_RED:
                continue
            match = classify(
                search_window(flat, river.cells[cell]),
                self._reds,
                size=face_size(seat=SEATS[position], row=cell // COLUMNS),
            )
            if match.margin >= CONFIDENT[1]:
                known[cell] = ms_to_mjai(match.label)
                self._guess[position][cell] = known[cell]

    def _emit(self, position: int, known: list[str]) -> None:
        """把這一家看得見的河交給 :class:`RiverWatcher`,新的捨牌送進 tracker。"""
        for discard in self._rivers.observe(position, known):
            self._turn_discards.append(
                (
                    position,
                    None if discard.pai == UNKNOWN else discard.pai,
                    self._guess[position].get(discard.index),
                    discard.serial,
                )
            )
            if discard.pai != UNKNOWN:
                self._table.handle(Dahai(actor=position, pai=discard.pai, tsumogiri=False))
        self._update_readings(position, known)

    def _update_readings(self, position: int, known: list[str]) -> None:
        """重讀之後的答案記回去,事件流會用。

        河少一張而鳴牌還沒確定的那一輪,格子與第幾張對不上,就先不記。已經有信心的
        不會被蓋掉 —— 與 ``RiverWatcher._fill`` 同一個理由。
        """
        serials = self._rivers.serials[position]
        if len(serials) != len(known):
            return
        for cell, (serial, pai) in enumerate(zip(serials, known, strict=True)):
            old = self._readings.get((position, serial))
            if old is not None and old[0] is not None:
                continue
            self._readings[(position, serial)] = (
                None if pai == UNKNOWN else pai,
                self._guess[position].get(cell),
            )

    def _look_for_sideways(
        self, position: int, flat: np.ndarray, river: River, known: list[str]
    ) -> int | None:
        """找橫放的立直宣言牌 —— 只問**可能是**的那幾格。

        宣言牌一定是認不出來的那幾格之一(實測 221/221 都過不了門檻),
        而且一定不是當列最後一張(最後一張沒有後續可驗)。候選通常是空的,
        那時候整個偵測不花錢 —— 不然每一家每一輪要 460 毫秒。
        """
        assert self._templates is not None
        filled = set(river.filled)
        candidates = [
            k
            for k, pai in enumerate(known)
            if pai == UNKNOWN and k in filled and (k // COLUMNS * COLUMNS + COLUMNS - 1) != k
            and (k + 1) in filled
        ]
        if not candidates:
            return None
        found = find_sideways(
            flat,
            self._templates,
            river=river,
            candidates=candidates,
            scale=DEFAULT_CORNERS[SEATS[position]].sideways_scale,
        )
        if found is None:
            return None
        self._sideways[position] = found
        # 自己立直看點數就好,這裡的誤判代價太大:實測手上的牌被當成立直,Mortal 之後只會回摸切。
        if position != 0:
            serials = self._rivers.serials[position]
            self._declare_reach(position, serials[found] if found < len(serials) else None)
        # 格線整個挪過,這一家要重認。
        known.clear()
        return found

    # ------------------------------------------------------------------ 手牌

    def _read_hands(
        self,
        hands: Sequence[tuple[tuple[str, ...], str | None]],
        image: np.ndarray,
        table: Rect,
    ) -> None:
        """這一輪收到的手牌交給 :class:`HandWatcher`。

        **牌局進行中才收。** 開局前的介面實測被讀成一副手牌,而結算畫面上留著的是
        上一局的手牌。桌上有牌,或余牌是開局的數字,才算進行中。
        """
        if not hands or self._digits is None:
            return
        if self._kyoku.empty and read_wall(image, table, self._digits) not in _OPENING_WALL:
            return
        for tiles, drawn in hands:
            mjai = [ms_to_mjai(tile) for tile in tiles]
            self._seen_hand = (mjai, None if drawn is None else ms_to_mjai(drawn))
            self._seen_at = self._turn_clock()
            self._own.observe(
                *self._seen_hand,
                thrown=self._own_thrown(),
                called=sum(self._melds.groups[1:]),
            )

    def _catch_up_own(self) -> None:
        """河裡比手牌多一張,而且等過手牌了:補上那次太快的摸切。"""
        discards = sum(move.move is Move.DISCARD for move in self._own.moves)
        thrown = self._own_thrown()
        if discards != self._own_counted:
            self._own_counted = discards
            self._kamicha_then = self._rivers.thrown(_KAMICHA)
        # 摸牌一定接在上家打牌之後。沒輪到自己就不會有摸切:立直的白光掃過自己的河會多讀出一張牌,
        # 照補的話憑空多一手,實測手牌之後每一手都晚一輪,最後余牌對不上,那一局 40 手沒有建議。
        called = bool(self._own.moves) and self._own.moves[-1].move is Move.CALL
        passed = not discards or called or self._rivers.thrown(_KAMICHA) > self._kamicha_then
        self._own_behind = self._own_behind + 1 if thrown > discards and passed else 0
        # 要的是手牌還沒交代的那一張,不是河裡最新的那張:手牌可能已經又落後一手。
        pai, guess = self._readings.get((0, discards), (None, None))
        tile = pai or guess
        patience = _OWN_PATIENCE_HELD if tile in self._own.hand else _OWN_PATIENCE
        if self._own_behind >= patience and self._own.catch_up(thrown, tile):
            self._own_behind = 0

    def _own_thrown(self) -> int:
        """自己打過幾張。河裡消失的牌,別家鳴牌沒有這麼多的話,多的那幾張本來就不是牌。"""
        thrown = self._rivers.thrown(0)
        gone = thrown - len(self._rivers.rivers[0])
        return thrown - max(gone - sum(self._melds.groups[1:]), 0)

    # ------------------------------------------------------------------ 副露

    def _read_melds(self, image: np.ndarray, table: Rect, rivers: Sequence[River]) -> None:
        """量四家的副露長度,長了一組就記一組。

        自己也讀:放銃分析用不到,但事件流要靠「誰鳴了」排出出牌順序。

        四家的牌河有一半以上讀不到(不連續)時,這一輪的副露長度都不採信:
        立繪與結算面板蓋住時,幾家的河**一起**壞掉。蓋住的那幾秒如果剛好讀到
        兩次一樣的假長度,就會變成一組假的副露 —— 而副露只增不減,錯了收不回來。

        不看「那一家自己的河」:對家河的第三列常常不連續(``docs/cv_mjai.md``
        §3.6),牌譜實測那樣會把對家一整段的副露擋掉(22 張)。
        """
        assert self._templates is not None
        self._collect_meld_faces()
        now = self._turn_clock()
        if now - self._meld_read < _MELD_GAP:
            return
        self._meld_read = now
        if (
            # 局還沒開始。段位場的大廳實測被讀成上家兩組副露,開局時又不會重設,
            # 那局上家真正的前兩組就被這個假的長度吃掉了。
            self._kyoku.empty
            or sum(river.is_plausible() for river in rivers) < _MELD_QUORUM
            # 還沒有人打過牌就不可能有人鳴牌。CPU 對局實測開局動畫讓下家與上家
            # 同時多記一組(那時四家的河都是空的)。
            or not any(river.count for river in rivers if river.is_plausible())
        ):
            for position in range(len(SEATS)):
                self._melds.observe(position, None)
            return
        for position, seat in enumerate(SEATS):
            start = self._melds.length(position)
            lead, length = occupied_span(image, table, MELD_VIEWS[seat])
            fresh = self._melds.observe(position, length)
            self._turn_melds.extend([position] * fresh)
            step = (length - start) / fresh if fresh else 0.0
            # 第一組的長度含開頭的空白。Steam 全螢幕自己那家空白比較大,碰會量成槓。
            blank = max(0.0, lead - _MELD_LEAD) if start == 0 else 0.0
            if fresh and step - blank / fresh >= KAN_WIDTH * GROUP_LENGTH[seat]:
                self._turn_kans.append(position)
            melds = self._table.players[position].melds
            for index in range(fresh):
                # 組數**現在**就算數(副露威脅不等牌面),牌面認好了再填進來。
                melds.append(guess_meld(()))
                future = self._faces.submit(
                    read_meld_faces,
                    image,
                    table,
                    seat,
                    self._templates,
                    start=start + index * step,
                    end=start + (index + 1) * step,
                    # 一次長了兩組:前一組後面緊接著下一組,尾端不能往外放寬。
                    open_end=index == fresh - 1,
                )
                self._pending_faces.append(
                    _PendingFaces(self._table, position, len(melds) - 1, future)
                )

    def _collect_meld_faces(self) -> None:
        """把認好的牌面填回去。

        為什麼不在這一輪直接認:一組約 0.5 秒,而剛有人鳴牌、下一張打出來
        的牌正是最需要危險度的時候。:func:`cv2.matchTemplate` 會放開 GIL,
        ``_faces`` 那條執行緒與這條真的是並行的。

        **只認一次。** 試過沒認全的組隔幾秒拿新畫面再認、結果併起來:兩場牌譜
        八個視角一組都沒有多認出來。漏的是系統性的(上家太糊、跨框的誤讀讓
        整組作廢),不是時機 —— ``MeldWatcher`` 本來就等長度穩定才確認。

        局換了就丟掉 —— 上一局的副露不該出現在這一局。認失敗就留著未知,
        跟認不出來一樣。
        """
        pending = []
        for job in self._pending_faces:
            if job.table is not self._table:
                job.future.cancel()
                continue
            if not job.future.done():
                pending.append(job)
                continue
            try:
                faces = job.future.result()
            except Exception as exc:  # noqa: BLE001 - 認不出來就是未知,不要讓整輪失敗
                logger.warning("副露牌面認不成:%s", exc)
                continue
            self._table.players[job.position].melds[job.index] = guess_meld(
                (normalize_red(ms_to_mjai(face)) for face in faces.tiles),
                repeated=(normalize_red(ms_to_mjai(face)) for face in faces.repeated),
            )
            if faces.red_suits:
                self._meld_reds[(job.position, job.index)] = faces.red_suits
        self._pending_faces = pending

    # ------------------------------------------------------------------ 寶牌

    def _read_dora(self, image: np.ndarray, table: Rect) -> None:
        """讀寶牌面板。

        **這一局開始了才讀。** 面板在結算畫面上還是上一局的,而收過的指示牌不會再看:實測五次開局
        寶牌讀錯,五次都是上一局那張,那一局 Mortal 從頭到尾看錯寶牌。條件與手牌相同,桌上有牌
        或余牌是開局的數字,94 局全對,最晚開局 3 秒讀到。
        """
        assert self._templates is not None
        if self._rois is None or self._digits is None:
            return
        if self._kyoku.empty and read_wall(image, table, self._digits) not in _OPENING_WALL:
            return
        try:
            panel = (
                self._rois.crop(image, _DORA_ROI)
                if self._panel is None
                else image[self._panel.as_slice()]
            )
        except (MissingRoiError, ValueError):
            return
        pixels = panel.tobytes()
        if pixels == self._dora_seen:
            return
        self._dora_seen = pixels
        for fresh in self._dora.observe(read_dora(panel, self._templates)):
            marker = ms_to_mjai(fresh.indicator)
            self._table.handle(Dora(dora_marker=marker))
            self._dora_marks.append((len(self.steps), marker))

    # ------------------------------------------------------------------ 其他

    def _declare_reach(self, position: int, serial: int | None = None) -> None:
        """記下這一家立直了。橫放牌與點數兩條路都會走到這裡,只算一次。

        Args:
            serial: 宣言牌是他的第幾張捨牌。橫放牌知道確切是哪一張;點數只知道
                「剛剛」,先當成最後一張,橫放牌之後找到會蓋過去。
        """
        if position == 0:
            if self._own_reach is None:
                thrown = sum(move.move is Move.DISCARD for move in self._own.moves)
                self._own_reach = thrown - 1
        elif serial is not None:
            self._reach_marks[position] = serial
        elif position not in self._reach_marks:
            serials = self._rivers.serials[position]
            self._reach_marks[position] = serials[-1] if serials else 0
        if self._table.players[position].reach:
            return
        self._table.handle(Reach(actor=position))
        self.status.say(f"{SEATS[position]} 立直了")

    def _mark_declared(self) -> None:
        """點數說有人立直了,宣言牌就是標記的那一張。不必再等橫放偵測。

        :func:`find_sideways` 要靠「挪了之後後面的牌認得出來」拍板,所以宣言牌是當列最後一張
        時永遠回 ``None``。它當初的理由是那時只錯自己一張,但那一張正是立直者的現物:實測
        一場 20 張讀錯的牌,9 張是宣言牌,4 張是同一列被推歪的牌。點數的判斷實測 13/13。

        直放認得出來就不是橫放的牌:宣言牌 221 格都過不了信心門檻。那多半是點數讀得晚,標記
        落在下一張上,交回給 :func:`find_sideways`。
        """
        for position, serial in self._reach_marks.items():
            if self._sideways[position] is not None:
                continue
            serials = self._rivers.serials[position]
            known = self._known[position]
            if serial not in serials or len(serials) != len(known):
                continue
            cell = serials.index(serial)
            # 橫放的牌比一格寬,後面那格也讀得到牌,點數一變標記就落在那一格上。實測宣言牌因此被當成
            # 直放,同一列後面全部讀歪。
            if cell - 1 in self._leaning[position] and known[cell - 1] == UNKNOWN:
                cell -= 1
            if known[cell] == UNKNOWN:
                self._sideways[position] = cell
                # 下一輪用挪過的格線整家重認,宣言牌也在那時候用旋轉模板讀。
                known.clear()

    def _read_round(self, image: np.ndarray, table: Rect, rivers: Sequence[River]) -> None:
        """這一局還不知道是哪一局的話,讀 HUD。

        用視角當座位(自己是 0),所以莊家直接就是 :attr:`Hud.dealer`。役牌只看
        相對莊家的位置,旋轉不影響 —— 而畫面這條路終於有場風與自風了,之前
        只有三元牌。

        **桌上還沒有牌就不讀。** 換局的空檔 HUD 正在更新:實測讀到過新的點數
        配舊的局數(東 3 局 4 本場的點數已經是下一局的),兩次一樣、總和也對,
        檢查碼擋不住。第一張牌出現的時候它一定已經換完了。

        讀穩之後還是每一輪都讀(1.5 毫秒):點數變了就是有人立直。

        **自己是莊家時等不到第一張牌**:Mortal 要在打第一張之前回答。所以配牌
        讀到了、余牌也是開局的數字,就算開始了 —— 手牌發下來的時候 HUD 已經換完。
        """
        assert self._digits is not None
        if not any(river.count for river in rivers if river.is_plausible()) and (
            not self._own.tehai or read_wall(image, table, self._digits) not in _OPENING_WALL
        ):
            return
        hud = read_hud(image, table, self._digits, panel=self._panel)
        if self._round.settled is not None:
            for position in self._reach.observe(hud):
                self._declare_reach(position)
            return
        settled = self._round.observe(hud)
        if settled is not None and settled.dealer is not None:
            self._table.set_round(settled.bakaze, settled.dealer)
            self._reach.start(settled)

    def _same_round(self, image: np.ndarray, table: Rect, rivers: Sequence[River]) -> bool:
        """桌面空了,但 HUD 還是這一局。見 :mod:`mia.tracker.kyoku`。

        讀不到局數也算沒變:多等一輪無害,而假的換局會把整局的進度清掉。
        """
        settled = self._round.settled
        if settled is None or self._digits is None or any(river.count for river in rivers):
            return False
        hud = read_hud(image, table, self._digits, panel=self._panel)
        return all(
            seen is None or seen == known
            for seen, known in (
                (hud.bakaze, settled.bakaze),
                (hud.kyoku, settled.kyoku),
                (hud.honba, settled.honba),
            )
        )

    def _load_digits(self) -> bool:
        if self._digits is None:
            try:
                self._digits = HudDigits.load()
            except (FileNotFoundError, ValueError) as exc:
                self.status.say(f"載不到 HUD 模板:{exc}")
                return False
        return True

    def _read_turns(self, image: np.ndarray, table: Rect) -> None:
        """這一輪的捨牌、副露、余牌交給 :class:`TurnTracker`,排出先後順序。

        莊家要等 HUD 讀穩才知道,所以開局那幾輪先存著。
        """
        discards, melds, kans = self._turn_discards, self._turn_melds, self._turn_kans
        self._turn_discards, self._turn_melds, self._turn_kans = [], [], []
        if self._digits is None or self._kyoku.empty:
            return
        discards, melds = self._own_first(discards, melds)
        wall = read_wall(image, table, self._digits)
        self._wall = wall if wall is not None else self._wall
        moved = sum(move.move is Move.DISCARD for move in self._own.moves)
        # 河裡已經比手牌多一張時,這副手牌是舊的,不是又輪到自己。鳴完打牌常常這樣:手牌還是鳴牌
        # 之後那一副,打掉的牌先到了河裡,這時不等副露,鳴牌就排成了摸牌。
        own_turn = (
            bool(self._own.hand)
            and len(self._own.hand) % 3 == 2
            and self._own_thrown() <= moved
        )
        now = self._turn_clock()
        if self._turns is None:
            self._turn_buffer.append((now, discards, melds, kans, wall))
            settled = self._round.settled
            if settled is None or settled.dealer is None:
                return
            self._turns = TurnTracker(settled.dealer)
            for tick, buffered, called, wide, seen in self._turn_buffer:
                self.steps += self._order(tick, buffered, called, seen, wide)
            self._turn_buffer = []
            return
        self.steps += self._order(now, discards, melds, wall, kans, hurry=own_turn)

    def _own_first(
        self, discards: Sequence[_Seen], melds: Sequence[int]
    ) -> tuple[list[_Seen], list[int]]:
        """自己打的牌與鳴牌,手牌一看到就交給排序,不等河裡那張飛完。

        自己的牌動畫最慢,要一秒以上才落定。排序是照輪替的,自己沒排出來,下家、對家打的牌就全部
        排在後面,吃碰要問的那幾張實測晚到 3 秒。河裡之後看到的同一張不再算一次。
        """
        kept = [d for d in discards if d[0] != 0 or self._own_drops.seen()]
        moved = [move.tiles[0] for move in self._own.moves if move.move is Move.DISCARD]
        kept += [(0, moved[n], None, n) for n in self._own_drops.tell(len(moved))]
        wanted = [position for position in melds if position != 0 or self._own_calls.seen()]
        called = sum(m.move is Move.CALL and len(m.tiles) == _OPEN_CALL for m in self._own.moves)
        wanted += [0 for _ in self._own_calls.tell(called)]
        return kept, wanted

    def _order(
        self,
        tick: float,
        discards: Sequence[_Seen],
        melds: Sequence[int],
        wall: int | None,
        kans: Sequence[int],
        *,
        hurry: bool = False,
    ) -> list[Step]:
        assert self._turns is not None
        if len(discards) >= _JOINED_LATE:
            # 余牌要跟著減才是真的牌。和了之後的結算畫面會一次讀出一排假牌,余牌不會動。
            drawn = _OPENING_WALL[0] - (wall if wall is not None else _OPENING_WALL[0])
            if (drawn - self._turns.drawn) * 2 < len(discards):
                return []
            return self._turns.jump(tick, discards, wall)
        return self._turns.tick(tick, discards, melds, wall, hurry=hurry, kans=kans)

    def _turn_clock(self) -> float:
        """:class:`TurnTracker` 的「輪」,一秒算一輪。

        **不能數讀了幾次。** 手牌一變就會提早讀,動畫中一秒讀好幾次,實測四輪最短 0.8 秒,而副露
        要 2 ~ 4 秒才確認,自己鳴牌就被排成摸牌,事件流停在那裡直到局尾。沒有跑 :meth:`run` 時
        (測試直接叫 :meth:`tick`)沒有時間可言,每叫一次算一輪。
        """
        if self._began is None:
            return float(self.reads)
        return (time.monotonic() - self._began) / max(self._interval, 0.001)

    @property
    def own_moves(self) -> tuple[OwnMove, ...]:
        """自己的動作,給除錯記錄用。"""
        return tuple(self._own.moves)

    @property
    def round(self) -> Hud | None:
        """這一局的開局資訊。HUD 還沒讀穩就是 ``None``。"""
        return self._round.settled

    def stream(self) -> Stream:
        """這一局到目前為止的事件流。見 :mod:`mia.tracker.stream`。"""
        settled = self._round.settled
        if settled is None:
            return Stream()
        log = self._kyoku_log(settled)
        stream = compose(log)
        stream = replace(stream, doubt=self._doubt(stream))
        if not self._stuck(stream) or not self._resync(log):
            return stream
        stream = compose(self._kyoku_log(settled))
        return replace(stream, doubt=self._doubt(stream))

    def _kyoku_log(self, settled: Hud) -> KyokuLog:
        return KyokuLog(
            hud=settled,
            tehai=self._own.tehai,
            doras=list(self._dora_marks),
            steps=list(self.steps),
            melds=[list(player.melds) for player in self._table.players],
            meld_reds=dict(self._meld_reds),
            reaches=self._reaches(),
            own_reach=self._own_reach,
            moves=list(self._own.moves),
            readings=dict(self._readings),
            next_up=self._turns.next_up if self._turns is not None else None,
        )

    def _stuck(self, stream: Stream) -> bool:
        """事件流停住夠久了。"""
        if not stream.stalled and self._own.stuck < _HAND_LOST:
            self._stalled_at = None
            return False
        now = self._turn_clock()
        if self._stalled_at is None:
            self._stalled_at = now
        # 停住之後要再讀到手牌才算數。局尾手牌被結算畫面蓋住,河裡卻還有最後一張,這時候重寫沒有
        # 意義:沒有新的手牌可以依據,而那一局已經結束了。
        fresh = self._seen_at is not None and self._seen_at > self._stalled_at
        return fresh and now - self._stalled_at >= _STALL_PATIENCE

    def _resync(self, log: KyokuLog) -> bool:
        """用畫面上現在的手牌重寫自己的動作。重寫得出來回 ``True``。"""
        self._stalled_at = None
        if self._seen_hand is None or not self._own.tehai:
            return False
        tiles, drawn = self._seen_hand
        fresh = resynced(log, tiles, drawn)
        if fresh is None:
            return False
        logger.info("事件流停住了,以現在的手牌重寫自己的動作")
        self._own.adopt(*fresh, tiles, drawn)
        # 重寫出來的動作都來自排序器,它早就知道了。
        self._own_drops.tell(sum(move.move is Move.DISCARD for move in fresh[1]))
        return True

    def _reaches(self) -> list[tuple[int, int]]:
        """立直落在哪一張。宣言牌多讀出來的那格要換回真的那張。"""
        doubled = self._turns.doubled if self._turns is not None else {}
        return [(p, doubled.get((p, s), s)) for p, s in self._reach_marks.items()]

    def _doubt(self, stream: Stream) -> str | None:
        """事件流還對得上畫面嗎。對不上就說為什麼。"""
        if self._own.stuck >= _HAND_LOST:
            return "手牌對不上"
        if self._wall is not None and stream.events:
            drawn = sum(isinstance(event, Tsumo) for event in stream.events)
            if abs(_OPENING_WALL[0] - self._wall - drawn) >= _WALL_LOST and not stream.stalled:
                return "余牌對不上"
        return None

    def _new_kyoku(self, *, keep_hand: bool = False) -> None:
        """清掉所有跟局有關的狀態。

        **上一局的安全牌在這一局毫無意義**,而留著不會有任何症狀,
        只會安靜地把危險牌說成安全 —— 與 ``TableTracker._reset`` 同一個理由。

        Args:
            keep_hand: 配牌留著。開局訊號要等第一張牌,而自己是莊家時,那一張
                就是自己打的:配牌與第一次摸牌都已經收過了。
        """
        self._table = TableTracker()
        self._table.seat = 0
        self._rivers.reset()
        self._dora.reset()
        self._melds.reset()
        self._round.reset()
        self._reach.reset()
        self.kyoku += 1
        if not keep_hand:
            self._own.reset()
        self._own_behind = 0
        self._own_drops = _OwnTold()
        self._own_calls = _OwnTold()
        self._own_counted = 0
        self._kamicha_then = 0
        self._dora_marks = []
        self._meld_reds = {}
        self._reach_marks = {}
        self._own_reach = None
        self._readings = {}
        self._turns = None
        self._turn_buffer = []
        self._turn_discards = []
        self._turn_melds = []
        self._turn_kans = []
        self.steps = []
        self._known = [[] for _ in SEATS]
        self._guess = [{} for _ in SEATS]
        self._sideways = [None for _ in SEATS]
        self._haku = [None for _ in SEATS]
        self._landing = [{} for _ in SEATS]
        self._still = [set() for _ in SEATS]
        self._clear = [False for _ in SEATS]
        self._leaning = [set() for _ in SEATS]
        self._unsteady = [set() for _ in SEATS]
        self._cells = {}
        self._dora_seen = b""
        self._seen_hand = None
        self._seen_at = None
        self._stalled_at = None
        self._last = None
        self._wall = None

    def _publish(self, hand: Sequence[str]) -> None:
        """算一次並投出去 —— 但只在結果真的變了的時候。

        與 :meth:`~mia.live.danger.DangerWorker._publish` 同一套理由,包括
        「兩個都要比」:只比 report 會漏掉「別人打掉我要的進張但危險度沒變」。
        """
        if not hand:
            return
        mjai = [ms_to_mjai(t) for t in hand]
        report = assess(mjai, self._table)
        seen = self._table.visible(mjai)
        if (report, seen) == self._last:
            return
        self._last = (report, seen)
        self.posts += 1
        self._bus.post(Dangers(report, seen))

    def __repr__(self) -> str:
        return (
            f"<TableWorker alive={self.status.alive} reads={self.reads} "
            f"posts={self.posts} 局={self._kyoku.started}>"
        )


class TableLink:
    """擷取那條路與牌桌那條路之間的轉接點。

    為什麼需要它:兩個功能**各自開關**,而
    :class:`~mia.live.runtime.Feature` 的模型是「開啟就造一個新的 worker」。
    所以 :class:`~mia.live.vision.VisionWorker` 不能直接握著
    :class:`TableWorker` —— 它握到的那個隨時會被丟掉。

    這個物件在整個 app 的生命週期裡只造一次,誰在就轉給誰,沒人在就丟掉。
    丟掉是對的:畫面每秒來十五張,而**牌河不需要那麼快**。
    """

    def __init__(self) -> None:
        # 放銃分析與 AI 建議各有一個 worker,兩個都開時畫面要送給兩邊。
        self._workers: tuple[TableWorker, ...] = ()

    def attach(self, worker: TableWorker) -> None:
        self._workers = (*self._workers, worker)

    def detach(self, worker: TableWorker) -> None:
        self._workers = tuple(w for w in self._workers if w is not worker)

    @property
    def attached(self) -> bool:
        return bool(self._workers)

    def offer_frame(self, image: np.ndarray, calibration: Calibration) -> None:
        for worker in self._workers:
            worker.offer_frame(image, calibration)

    def offer_hand(self, tiles: Sequence[str], drawn: str | None = None) -> None:
        for worker in self._workers:
            worker.offer_hand(tiles, drawn)
