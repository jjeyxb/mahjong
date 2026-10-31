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

全套三秒,15 fps 的迴圈完全吃不下。但**牌河不需要 15 fps** —— 一秒讀一次
綽綽有餘。所以這裡是一條獨立的執行緒,擷取那條路把最新的畫面丟進來,
這邊照自己的節奏吃。

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

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace

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
    ReachWatcher,
    RiverWatcher,
    RoundWatcher,
    Step,
    Stream,
    TurnTracker,
    compose,
    guess_meld,
)
from mia.utils.geometry import Rect
from mia.utils.logging import logger
from mia.vision.hud import Hud, HudDigits, read_hud, read_wall
from mia.vision.roi import MissingRoiError, RoiSet
from mia.vision.tiles.classify import DEFAULT_SKIN, MIN_BRIGHTNESS, TemplateSet
from mia.vision.tiles.dora import read_dora
from mia.vision.tiles.melds import DEFAULT_VIEWS as MELD_VIEWS
from mia.vision.tiles.melds import (
    GROUP_LENGTH,
    MeldFaces,
    occupied_length,
    read_meld_faces,
)
from mia.vision.tiles.river import (
    COLUMNS,
    CONFIDENT,
    DEFAULT_CORNERS,
    SEATS,
    River,
    classify_river_tile,
    face_size,
    find_sideways,
    read_river,
    search_window,
)

__all__ = ["TableLink", "TableWorker"]

#: 兩次讀取之間至少隔多久。牌河每家約五秒才多一張 —— 一秒一次綽綽有餘,
#: 而且讓出 CPU 給擷取那條路(它要維持 15 fps)。
_INTERVAL = 1.0

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

#: 連續幾副不同的手牌解釋不了,就算事件流對不上了。見 ``stream``。
_HAND_LOST = 3

#: 副露一組長了這麼多個組長以上就是槓。吃碰是 1.0,槓實測 1.13 以上。
_KAN_WIDTH = 1.1

#: 排出來的摸牌數與余牌差到幾張,就算事件流對不上了。排序本來就會落後幾步,
#: 而加槓的嶺上看不見。
_WALL_LOST = 5

#: 一輪湧進這麼多張捨牌,代表是中途接上、一次看到滿河。出牌順序已經無從得知,
#: 這一局不出事件流。CPU 對局一秒最多兩個動作,實測一輪最多 3 張。
_JOINED_LATE = 4


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
        self._skin = skin
        self._interval = interval
        self._lock = threading.Lock()
        self._frame: tuple[np.ndarray, Calibration] | None = None
        self._hand: tuple[str, ...] = ()
        # 兩輪之間手牌可能變了好幾次(摸、打),一次都不能漏,所以是佇列。
        self._hands: list[tuple[tuple[str, ...], str | None]] = []
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._templates = templates
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
        self._turn_buffer: list[tuple[int, list[_Seen], list[int], list[int], int | None]] = []
        self._turn_discards: list[_Seen] = []
        self._turn_melds: list[int] = []
        self._turn_kans: list[int] = []
        self._joined_late = False
        self.steps: list[Step] = []
        self._own = HandWatcher()
        self._dora_marks: list[tuple[int, str]] = []
        self._reach_marks: dict[int, int] = {}
        self._own_reach: int | None = None
        self._readings: dict[tuple[int, int], tuple[str | None, str | None]] = {}
        self._melds = MeldWatcher(units=tuple(GROUP_LENGTH[seat] for seat in SEATS))
        # 認牌面慢(一組 4 ~ 5 秒),放到另一條執行緒。見 _read_melds。
        self._faces = ThreadPoolExecutor(max_workers=1, thread_name_prefix="meld-faces")
        self._pending_faces: list[_PendingFaces] = []
        self._known: list[list[str]] = [[] for _ in SEATS]
        # 每家每一格的最佳猜測,沒信心的也留著:Mortal 不收 ?。
        self._guess: list[dict[int, str]] = [{} for _ in SEATS]
        self._sideways: list[int | None] = [None for _ in SEATS]
        # 每家「上一輪讀成白、還沒收下」的那一格。見 _read_seat。
        self._haku: list[int | None] = [None for _ in SEATS]
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
        if self._templates is None:
            try:
                self._templates = TemplateSet.load(self._skin)
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
                self._wake.wait(max(0.0, self._interval - (time.monotonic() - started)))
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
        self.reads += 1
        if not self._load_digits():
            return
        if self._rois is None or not self._rois.matches(calibration):
            self._rois = RoiSet(self._config, calibration)

        table = calibration.table_rect
        flats = [DEFAULT_CORNERS[seat].rectify(image, table) for seat in SEATS]
        rivers = [read_river(flat) for flat in flats]

        signal = self._kyoku.observe(rivers)
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
                self._read_seat(position, flat, river)

        self._read_hands(hands, image, table)
        self._read_melds(image, table, rivers)
        self._read_dora(image)
        self._read_round(image, table, rivers)
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

    def _read_seat(self, position: int, flat: np.ndarray, river: River) -> None:
        """認一家的河,只認新的那幾格。"""
        assert self._templates is not None
        known = self._known[position]
        filled = river.filled
        if len(filled) == len(known):
            # 沒有新格子,不必重認 —— 但觀察照樣交出去:一口氣長了好幾張的
            # 河要靠**下一次**觀察確認才會發事件(過場動畫的白光,見
            # ``mia.tracker.rivers._JUMP``)。
            self._emit(position, known)
            return
        if len(filled) < len(known):
            # 被鳴走了一張 —— 後面整排往前遞補,快取整個失效。
            # 一場約九次,重認一家 300 毫秒,划得來。
            known.clear()

        sideways = self._sideways[position]
        if sideways is None:
            sideways = self._look_for_sideways(position, flat, river, known)
        if sideways is not None:
            scale = DEFAULT_CORNERS[SEATS[position]].sideways_scale
            river = read_river(flat, sideways=sideways, scale=scale)
            filled = river.filled

        size = face_size()
        fresh = [k for k in filled if k >= len(known)] if len(known) else list(filled)
        # 整條重認(剛接上、鳴牌或立直之後)不必防白光:那不是白光掃過的時機。
        incremental = bool(known)
        for cell in fresh:
            if cell in river.clipped:
                known.append(UNKNOWN)
                continue
            match = classify_river_tile(
                search_window(flat, river.cells[cell]), self._templates, size=size
            )
            confident = (
                match.score >= CONFIDENT[0]
                and match.margin >= CONFIDENT[1]
                and match.brightness >= MIN_BRIGHTNESS
            )
            pai = ms_to_mjai(match.label) if confident else UNKNOWN
            self._guess[position][cell] = ms_to_mjai(match.label)
            if incremental and pai == _HAKU and self._haku[position] != cell:
                # 白要看兩次才收。和了 / 立直的過場有一道白光橫過牌河,空白的
                # 牌面模板對上白光,讀成**有信心的白** —— 而這裡每一格只認一次,
                # 收下就永遠留著,變成一張假現物。重播校正素材:真的一索在白光
                # 掃過的那一刻打出來,被讀成白,上家因此 30 輪都有一張假現物。
                # 這一輪先停在這一格,下一輪重認;真的白只是晚一輪。
                self._haku[position] = cell
                break
            self._haku[position] = None
            known.append(pai)

        self._emit(position, known)

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
            self._own.observe(
                [ms_to_mjai(tile) for tile in tiles],
                None if drawn is None else ms_to_mjai(drawn),
                thrown=self._rivers.thrown(0),
            )

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
        if (
            sum(river.is_plausible() for river in rivers) < _MELD_QUORUM
            # 還沒有人打過牌就不可能有人鳴牌。Steam 實測開局動畫讓下家與上家
            # 同時多記一組(那時四家的河都是空的)。
            or not any(river.count for river in rivers if river.is_plausible())
        ):
            for position in range(len(SEATS)):
                self._melds.observe(position, None)
            return
        for position, seat in enumerate(SEATS):
            start = self._melds.length(position)
            length = occupied_length(image, table, MELD_VIEWS[seat])
            fresh = self._melds.observe(position, length)
            self._turn_melds.extend([position] * fresh)
            step = (length - start) / fresh if fresh else 0.0
            if fresh and step >= _KAN_WIDTH * GROUP_LENGTH[seat]:
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

        為什麼不在這一輪直接認:一組要 4 ~ 5 秒,而剛有人鳴牌、下一張打出來
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
        self._pending_faces = pending

    # ------------------------------------------------------------------ 寶牌

    def _read_dora(self, image: np.ndarray) -> None:
        assert self._templates is not None
        if self._rois is None:
            return
        try:
            panel = self._rois.crop(image, _DORA_ROI)
        except (MissingRoiError, ValueError):
            return
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
        hud = read_hud(image, table, self._digits)
        if self._round.settled is not None:
            for position in self._reach.observe(hud):
                self._declare_reach(position)
            return
        settled = self._round.observe(hud)
        if settled is not None and settled.dealer is not None:
            self._table.set_round(settled.bakaze, settled.dealer)
            self._reach.start(settled)

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
        if self._digits is None or self._kyoku.empty or self._joined_late:
            return
        if len(discards) >= _JOINED_LATE:
            self._joined_late = True
            self._turns = None
            self._turn_buffer = []
            return
        wall = read_wall(image, table, self._digits)
        self._wall = wall if wall is not None else self._wall
        own_turn = bool(self._own.hand) and len(self._own.hand) % 3 == 2
        if self._turns is None:
            self._turn_buffer.append((self.reads, discards, melds, kans, wall))
            settled = self._round.settled
            if settled is None or settled.dealer is None:
                return
            self._turns = TurnTracker(settled.dealer)
            for tick, buffered, called, wide, seen in self._turn_buffer:
                self.steps += self._turns.tick(tick, buffered, called, seen, kans=wide)
            self._turn_buffer = []
            return
        self.steps += self._turns.tick(
            self.reads, discards, melds, wall, hurry=own_turn, kans=kans
        )

    @property
    def round(self) -> Hud | None:
        """這一局的開局資訊。HUD 還沒讀穩就是 ``None``。"""
        return self._round.settled

    def stream(self) -> Stream:
        """這一局到目前為止的事件流。見 :mod:`mia.tracker.stream`。"""
        settled = self._round.settled
        if settled is None or self._joined_late:
            return Stream()
        stream = compose(
            KyokuLog(
                hud=settled,
                tehai=self._own.tehai,
                doras=list(self._dora_marks),
                steps=list(self.steps),
                melds=[list(player.melds) for player in self._table.players],
                reaches=list(self._reach_marks.items()),
                own_reach=self._own_reach,
                moves=list(self._own.moves),
                readings=dict(self._readings),
            )
        )
        return replace(stream, doubt=self._doubt(stream))

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
        self._dora_marks = []
        self._reach_marks = {}
        self._own_reach = None
        self._readings = {}
        self._turns = None
        self._turn_buffer = []
        self._turn_discards = []
        self._turn_melds = []
        self._turn_kans = []
        self._joined_late = False
        self.steps = []
        self._known = [[] for _ in SEATS]
        self._guess = [{} for _ in SEATS]
        self._sideways = [None for _ in SEATS]
        self._haku = [None for _ in SEATS]
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
