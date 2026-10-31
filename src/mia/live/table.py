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
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from mia.analysis.danger import assess
from mia.calibration.table import Calibration
from mia.config.models import RoiConfig
from mia.live.bus import Dangers, UpdateBus, WorkerStatus
from mia.mjai.events import Dahai, Dora, Reach
from mia.mjai.table import TableTracker
from mia.mjai.tiles import ms_to_mjai, normalize_red
from mia.tracker import (
    UNKNOWN,
    DoraWatcher,
    KyokuSignal,
    KyokuWatcher,
    MeldWatcher,
    RiverWatcher,
    RoundWatcher,
    guess_meld,
)
from mia.utils.geometry import Rect
from mia.utils.logging import logger
from mia.vision.hud import Hud, HudDigits, read_hud
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

    Note:
        **這條路的狀態比封包那條路少。** 沒有 ``start_kyoku`` 就不知道場風
        與莊家,所以役牌只剩三元牌 —— 那是 :class:`~mia.mjai.table.Player`
        的預設值,方向是保守的。HUD 上讀得到局數與莊家(``round_info``),
        但那還沒做。
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
    ) -> None:
        super().__init__(name="table", daemon=True)
        self._link = link
        self._bus = bus
        self._config = config
        self._rois: RoiSet | None = None
        self._skin = skin
        self._interval = interval
        self._lock = threading.Lock()
        self._frame: tuple[np.ndarray, Calibration] | None = None
        self._hand: tuple[str, ...] = ()
        self._stop = threading.Event()
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
        self._melds = MeldWatcher(units=tuple(GROUP_LENGTH[seat] for seat in SEATS))
        # 認牌面慢(一組 4 ~ 5 秒),放到另一條執行緒。見 _read_melds。
        self._faces = ThreadPoolExecutor(max_workers=1, thread_name_prefix="meld-faces")
        self._pending_faces: list[_PendingFaces] = []
        self._known: list[list[str]] = [[] for _ in SEATS]
        self._sideways: list[int | None] = [None for _ in SEATS]
        # 每家「上一輪讀成白、還沒收下」的那一格。見 _read_seat。
        self._haku: list[int | None] = [None for _ in SEATS]
        self._last: tuple | None = None

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

    def offer_hand(self, tiles: Sequence[str]) -> None:
        """收一副手牌(雀魂記法)。危險度要知道自己手上有什麼才算得出來。"""
        with self._lock:
            self._hand = tuple(tiles)

    def stop(self) -> None:
        self._stop.set()

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
                self._stop.wait(max(0.0, self._interval - (time.monotonic() - started)))
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
            self._frame = None
        if frame is None:
            # **不要安靜地 return。** 使用者回報過:只打開放銃分析的話畫面
            # 一直停在「等待對局…」,而這裡拿不到畫面就直接回去,沒有任何
            # 訊息 —— 看起來就是撥了開關什麼都沒發生。
            #
            # 現在 LiveRuntime 會一併打開畫面辨識(見 _ensure_vision_for),
            # 所以這句話只會在真的還沒送到第一幀時出現(找視窗、校正中)。
            self.status.say("等畫面辨識把牌桌送過來")
            return

        image, calibration = frame
        self.reads += 1
        if self._rois is None or not self._rois.matches(calibration):
            self._rois = RoiSet(self._config, calibration)

        table = calibration.table_rect
        flats = [DEFAULT_CORNERS[seat].rectify(image, table) for seat in SEATS]
        rivers = [read_river(flat) for flat in flats]

        signal = self._kyoku.observe(rivers)
        if signal is not KyokuSignal.NONE:
            self._new_kyoku()
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

        self._read_melds(image, table, rivers)
        self._read_dora(image)
        self._read_round(image, table, rivers)
        # **狀態寫在最後。** 寫在開頭的話會被這一輪後面的訊息蓋掉 ——
        # 實測「等讀到自己的手牌」被局邊界的「新的一局」洗掉了,而那正是
        # 使用者最需要看到前者的時候。狀態列要講的是這一輪**結束時**的狀態。
        #
        # 危險度是「我切這張會不會放銃」—— 沒有手牌就沒有問題可以回答。
        # 牌河照常讀下去,等手牌來了才算得出來。
        self.status.say("" if hand else "等讀到自己的手牌")
        self._publish(hand)

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
            if discard.pai != UNKNOWN:
                self._table.handle(Dahai(actor=position, pai=discard.pai, tsumogiri=False))

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
        self._table.handle(Reach(actor=position))
        self.status.say(f"{SEATS[position]} 立直了")
        # 格線整個挪過,這一家要重認。
        known.clear()
        return found

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
            step = (length - start) / fresh if fresh else 0.0
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
            self._table.handle(Dora(dora_marker=ms_to_mjai(fresh.indicator)))

    # ------------------------------------------------------------------ 其他

    def _read_round(self, image: np.ndarray, table: Rect, rivers: Sequence[River]) -> None:
        """這一局還不知道是哪一局的話,讀 HUD。

        用視角當座位(自己是 0),所以莊家直接就是 :attr:`Hud.dealer`。役牌只看
        相對莊家的位置,旋轉不影響 —— 而畫面這條路終於有場風與自風了,之前
        只有三元牌。

        **桌上還沒有牌就不讀。** 換局的空檔 HUD 正在更新:實測讀到過新的點數
        配舊的局數(東 3 局 4 本場的點數已經是下一局的),兩次一樣、總和也對,
        檢查碼擋不住。第一張牌出現的時候它一定已經換完了。
        """
        if self._round.settled is not None or not any(
            river.count for river in rivers if river.is_plausible()
        ):
            return
        if self._digits is None:
            try:
                self._digits = HudDigits.load()
            except (FileNotFoundError, ValueError) as exc:
                self.status.say(f"載不到 HUD 模板:{exc}")
                return
        settled = self._round.observe(read_hud(image, table, self._digits))
        if settled is not None and settled.dealer is not None:
            self._table.set_round(settled.bakaze, settled.dealer)

    @property
    def round(self) -> Hud | None:
        """這一局的開局資訊。HUD 還沒讀穩就是 ``None``。"""
        return self._round.settled

    def _new_kyoku(self) -> None:
        """清掉所有跟局有關的狀態。

        **上一局的安全牌在這一局毫無意義**,而留著不會有任何症狀,
        只會安靜地把危險牌說成安全 —— 與 ``TableTracker._reset`` 同一個理由。
        """
        self._table = TableTracker()
        self._table.seat = 0
        self._rivers.reset()
        self._dora.reset()
        self._melds.reset()
        self._round.reset()
        self._known = [[] for _ in SEATS]
        self._sideways = [None for _ in SEATS]
        self._haku = [None for _ in SEATS]
        self._last = None

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
        self._worker: TableWorker | None = None

    def attach(self, worker: TableWorker) -> None:
        self._worker = worker

    def detach(self, worker: TableWorker) -> None:
        """只有自己還在位子上才拔掉 —— 後來的那個不該被前一個的收尾清掉。"""
        if self._worker is worker:
            self._worker = None

    @property
    def attached(self) -> bool:
        return self._worker is not None

    def offer_frame(self, image: np.ndarray, calibration: Calibration) -> None:
        worker = self._worker
        if worker is not None:
            worker.offer_frame(image, calibration)

    def offer_hand(self, tiles: Sequence[str]) -> None:
        worker = self._worker
        if worker is not None:
            worker.offer_hand(tiles)
