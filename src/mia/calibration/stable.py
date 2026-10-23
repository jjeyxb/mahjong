"""多幀穩定化的牌桌校正。

為什麼需要這一層
----------------
:meth:`~mia.calibration.table.TableCalibrator.calibrate` 是**單幀**
啟發式:從四邊往內剝除整條顏色一致的帶狀區域。這個訊號會被畫面內容干擾 ——
和了動畫、角色立繪、暗轉、載入畫面都會讓它剝多或剝少。

實測一份 216 幀的錄影(同一個視窗、同一場對局、視窗尺寸從頭到尾沒變),
單幀校正產生了 **21 種不同的 ``table_rect``**:

===========================  ======  ==========
矩形                          幀數    寬高比
===========================  ======  ==========
``2560x1430@(0,174)``          108    1.7902  ← 正確
``2560x1429@(0,174)``           61    1.7915
``2560x1440@(0,174)``           13    1.7778
``2302x1429@(5,174)``            5    1.6109  ← 寬度少 258 px
``2564x1692@(0,0)``              4    1.5154  ← 完全沒剝掉瀏覽器工具列
``2560x1142@(0,174)``            1    2.2417
…(另有 15 種)
===========================  ======  ==========

約 **16% 的幀給出明顯錯誤的答案**。ROI 全部相對 ``table_rect`` 正規化,
寬度從 2560 掉到 2302 就讓每個 ROI 整體位移約 10%(≈100 px)—— 比一張牌還寬。

原本的作法為什麼不夠
--------------------
錄製迴圈本來就有快取:第一幀校正完就鎖住,只在影像尺寸改變時重算。這確實
避免了逐幀跳動,但把問題換成了一場賭博 —— 有 16% 的機率把錯的鎖進整場對局,
而且遊戲剛啟動時常常正停在載入畫面或選單,**那正是剝除最容易失敗的時候**。

作法
----
連續取樣,丟掉明顯不合理的候選,再對剩下的取**逐分量中位數**。

中位數而不是眾數,是因為要同時處理兩種雜訊:少數的離群值(那 16%),以及
多數樣本之間 ±1 px 的抖動。眾數會被後者拆散 —— 上表裡 1430 佔 108 幀、
1429 佔 61 幀,它們其實是同一個答案,眾數卻只會取到其中一個。中位數兩者都吃。

丟掉候選的條件刻意只有兩個(偵測失敗、寬高比離譜),而不是「有警告就丟」——
正確答案 1.7902 本身就偏離 16:9 達 0.7%,把示警當成拒絕條件會把對的丟掉。

丟太多的時候要**拒絕鎖定**
--------------------------
上面那套的前提是「大多數候選是對的,少數是雜訊」。這個前提會破 —— 實測
900x593 的視窗下,剝除停在瀏覽器工具列裡,91% 的候選因寬高比 1.56 被丟掉,
而**通過檢查的那 9% 恰恰是剝過頭、剝進遊戲畫布裡的幀**。寬高比檢查於是
反過來在挑錯的答案,中位數取的是一組有系統性偏差的離群值。

那次鎖進去的矩形位移了約 0.7 張牌寬,手牌辨識從 96% 掉到 29%,**整場錄影
沒有任何錯誤訊息**。所以現在的規則是:近期丟棄的比採用的多就不鎖,
讓上層當場說「校正失敗」。詳見 :mod:`mia.calibration.canvas`。

鎖定之後還要繼續複查
--------------------
上面兩道防線擋的都是「候選看起來不對」。但有一種情況是候選看起來**完全正常**
而答案仍然是錯的:鎖定時畫面根本還不是牌桌。

實測 ``data/recordings/20260801-112437``:錄製開頭遊戲正在重連,前十幾幀是
黑色載入畫面。剝除在那上面剝得很乾淨,給出 ``1708x968@(69,218)``,寬高比
1.7645 —— 離 16:9 只差 0.7%,兩道防線一道都沒響(採用 15 個候選、丟棄 0 個)。
那個矩形被鎖進整場,而牌桌出現之後真正的答案是 ``1798x1152@(1,27)``。

後果與前面那次一樣:手牌的槽位整排在飄(``pitch`` 是 ROI 寬度的比例,誤差
會累加,到第 13 張差了半張牌),四家牌河的角點也全部偏掉。**而且一樣沒有
任何錯誤訊息** —— 唯一的症狀是準確率難看。

所以鎖定不是終點。鎖定之後定期拿單幀校正跟鎖定值對一次,近期不一致的比
一致的多就解鎖重來。兩者分得很開(抽 122 幀):

=========================  ============  ==========
錄影                        不一致比例    偏差中位
=========================  ============  ==========
110639(鎖定值正確)          16%           10 px
112437(鎖定在載入畫面上)    97%          191 px
=========================  ============  ==========

16% 正是單幀校正本來的雜訊,多數決離它很遠。
"""

from __future__ import annotations

from collections import deque
from statistics import median

from mia.calibration.canvas import CanvasChoice
from mia.calibration.manual import TableRectChoice
from mia.calibration.table import Calibration, TableCalibrator
from mia.capture.base import Frame
from mia.config.models import CalibrationConfig
from mia.utils.geometry import NormRect, Rect, Size
from mia.utils.logging import logger

__all__ = ["StableCalibrator"]

#: 鎖定後,若各候選在任一分量上的全距超過短邊的這個比例,就在結果裡記一筆警告。
#: 不拒絕鎖定 —— 中位數本來就是為了在雜訊中給答案,但使用者該知道畫面很不穩。
_SPREAD_WARN_RATIO = 0.02

#: 鎖定後每隔幾幀複查一次。
#:
#: 不是每幀都查:複查要重跑一次剝除,而鎖定後本來就不用再算了。15 幀約一秒
#: (實測擷取 15 fps),所以複查大致是一秒一次。
#:
#: **這個值與 :data:`_VERIFY_WINDOW` 要一起看** —— 真正該管的是視窗涵蓋多少
#: **時間**,不是多少次。查得更密不會讓判斷更準,只會讓同一段動畫塞滿視窗。
_VERIFY_EVERY = 15

#: 複查的滑動視窗長度(次)。約一秒一次,所以這也約等於秒數。
#:
#: 為什麼要這麼長
#: --------------
#: 和了動畫、結算、配牌那段時間畫面上沒有牌桌邊界可剝,複查會連續不一致 ——
#: 那是**沒看到**,不是鎖錯了。實測 ``20260801-110639``(鎖定值正確)最長
#: 連續不一致 27 次,所以視窗必須明顯長過那個,不然一次和了就能把一個正確的
#: 鎖定解掉,而解錯的代價是整個辨識停擺。
#:
#: 掃過各種長度,取兩份錄影在各自最糟的視窗裡的不一致比例:
#:
#: ======  ====================  ====================
#: 視窗    110639(鎖定正確)      112437(鎖在載入畫面)
#: ======  ====================  ====================
#: 15      100%                  100%
#: 30      90%                   100%
#: 60      45%                   100%
#: **80**  **36%**               **100%**
#: 120     28%                   100%
#: ======  ====================  ====================
#:
#: 取 80:正確的那份離半數還有 14 個百分點,而錯誤的那份是滿分。視窗 60 的
#: 餘裕只剩 5 個百分點,太薄。代價是發現錯誤鎖定要約 80 秒 —— 對一場十分鐘
#: 的對局可以接受,而現狀是**永遠不會發現**。
_VERIFY_WINDOW = 80

#: 複查時,鎖定值與單幀答案在任一分量上差多少才算「不一致」,以短邊為單位。
#:
#: 與 :data:`_SPREAD_WARN_RATIO` 同值不是巧合:那裡問的是「各幀彼此差多少」,
#: 這裡問的是「現在跟當初差多少」,兩者是同一種容忍度。實測正確的鎖定偏差
#: 中位 10 px(門檻 32 px),錯誤的鎖定偏差中位 191 px(門檻 24 px)。
_VERIFY_TOLERANCE = 0.02

#: 判斷「丟太多」時只看最近這麼多次嘗試,單位是 ``stabilize_frames`` 的倍數。
#:
#: 用滑動視窗而不是累計,是為了**能夠復原**:遊戲還停在登入畫面或載入畫面時
#: 剝除本來就會一直失敗,那段時間累積的丟棄數不該一路跟著使用者到牌桌上。
#: 用累計的話,登入花了兩分鐘的人就再也鎖不上了。
_WINDOW_FACTOR = 4


class StableCalibrator:
    """蒐集多幀的單幀校正結果,取中位數後鎖定。

    典型用法是每擷取一幀就 :meth:`feed` 一次,在它回傳非 ``None`` 之前
    先不要開始辨識::

        stable = StableCalibrator(config.calibration)
        while True:
            frame = backend.capture(window)
            calibration = stable.feed(frame)
            if calibration is None:
                continue          # 還在蒐集,這幀先跳過
            ...

    鎖定之後 :meth:`feed` 會直接回傳同一份結果,不再重算 —— 除非影像尺寸
    改變(使用者縮放視窗、切全螢幕),那會自動重置並重新蒐集。
    """

    def __init__(
        self,
        config: CalibrationConfig | None = None,
        *,
        canvas: CanvasChoice | None = None,
        manual: TableRectChoice | None = None,
    ) -> None:
        self.config = config or CalibrationConfig()
        self._calibrator = TableCalibrator(self.config, canvas=canvas, manual=manual)
        self._samples: list[Rect] = []
        self._rejected = 0
        self._recent: deque[bool] = deque(
            maxlen=max(2, self.config.stabilize_frames * _WINDOW_FACTOR)
        )
        self._blocked: str | None = None
        self._result: Calibration | None = None
        self._size: Size | None = None
        self._checks: deque[bool] = deque(maxlen=_VERIFY_WINDOW)
        self._until_check = _VERIFY_EVERY
        #: 目前這個鎖定是在哪個手動框之下成立的。使用者重框之後要重算,
        #: 而「重框了沒」只有比對得出來 —— 那個格子不會通知任何人。
        self._manual: NormRect | None = None

    @property
    def canvas(self) -> CanvasChoice:
        """使用者選的畫布尺寸。改了之後要自己呼叫 :meth:`reset`。"""
        return self._calibrator.canvas

    @property
    def manual(self) -> TableRectChoice:
        """使用者框的牌桌矩形。**改了不必自己 reset** —— :meth:`feed` 每一幀
        都看一次,所以框完下一幀就生效。

        與 :attr:`canvas` 不同是刻意的:選畫布是在選單裡挑一個 preset,動作
        結束在選單上;框牌桌是對著畫面拖一個框,使用者期待的是「框完就好了」。
        """
        return self._calibrator.manual

    # ------------------------------------------------------------------ 狀態

    @property
    def result(self) -> Calibration | None:
        """已鎖定的校正;還在蒐集時為 ``None``。"""
        return self._result

    @property
    def locked(self) -> bool:
        return self._result is not None

    @property
    def progress(self) -> tuple[int, int]:
        """``(已接受的候選數, 需要的數量)``,給 UI 顯示進度用。"""
        return len(self._samples), self.config.stabilize_frames

    @property
    def failure(self) -> str | None:
        """為什麼鎖不上。還在正常蒐集(或已鎖定)時為 ``None``。

        這是**當下**的判斷,不是latch住的旗標:情況好轉(近期的候選開始
        通過)之後就會自己消失,見 :data:`_WINDOW_FACTOR`。
        """
        return None if self._result is not None else self._blocked

    def reset(self) -> None:
        self._samples.clear()
        self._rejected = 0
        self._recent.clear()
        self._blocked = None
        self._result = None
        self._size = None
        self._checks.clear()
        self._until_check = _VERIFY_EVERY
        self._manual = None

    # ------------------------------------------------------------------ 主流程

    def feed(self, frame: Frame) -> Calibration | None:
        """餵一幀進去。回傳已鎖定的校正,或 ``None`` 表示還在蒐集。"""
        # 使用者在 UI 上重框了牌桌 —— 立刻作廢,不要等複查。那是人明確下的
        # 指令,而複查最快也要八十秒;那八十秒裡畫面上會「框完了卻沒反應」,
        # 而那正是這個功能要解決的那種沉默。
        manual = self.manual.value
        if self._result is not None and manual != self._manual:
            logger.info("手動框選的牌桌矩形改成 {},重新校正", self.manual)
            self.reset()
        self._manual = manual

        if self._size is not None and frame.size != self._size:
            logger.info("影像尺寸由 {} 變為 {},重新校正", self._size, frame.size)
            self.reset()

        # 解鎖之後這一幀要繼續往下當成候選蒐集 —— 直接 return None 的話,
        # 畫面已經好了的情況下還要多等一幀才開始,沒有理由。
        if self._result is not None and self._still_valid(frame):
            return self._result

        self._size = frame.size
        candidate = self._calibrator.calibrate(frame)

        # 手動指定與畫布推導都是算出來的,不是猜的 —— 沒有「不穩定」可言,直接鎖定。
        if candidate.source in ("manual", "canvas"):
            self._result = candidate
            logger.info("牌桌校正已鎖定({}): {}", candidate.source, candidate)
            return self._result

        accepted = self._accepts(candidate)
        self._recent.append(accepted)
        if accepted:
            self._samples.append(candidate.table_rect)
        else:
            self._rejected += 1

        # 選了畫布卻推導失敗時,calibrate 會回 fallback 並附上原因。原封不動
        # 轉述給上層 —— 「畫布對不上」比「候選都被丟掉」精確得多。
        self._blocked = self._diagnose(candidate)
        if self._blocked is not None:
            return None

        if len(self._samples) < self.config.stabilize_frames:
            return None

        self._result = self._settle(frame)
        logger.info(
            "牌桌校正已鎖定: {}(採用 {} 個候選,丟棄 {} 個)",
            self._result,
            len(self._samples),
            self._rejected,
        )
        return self._result

    # ------------------------------------------------------------------ 內部

    def _still_valid(self, frame: Frame) -> bool:
        """已鎖定的校正現在還對嗎。不對就當場解鎖,回 ``False``。

        複查的理由見模組說明:前面兩道防線問的都是「候選看起來對不對」,
        而鎖定在載入畫面上的那次,候選看起來完全正常。唯一分得出來的證據是
        **牌桌出現之後,畫面說的跟當初鎖的不一樣**。

        手動指定與畫布推導不複查 —— 那兩個是算出來的,不是從畫面猜的,
        使用者講了什麼就是什麼。畫面對不上是使用者要知道的事,不是這裡該
        自作主張改掉的事。
        """
        assert self._result is not None
        if self._result.source in ("manual", "canvas"):
            return True

        self._until_check -= 1
        if self._until_check > 0:
            return True
        self._until_check = _VERIFY_EVERY

        matched = self._matches_lock(frame)
        if matched is None:
            # 這一幀連內容區都找不到(暗轉、全黑)。那是「沒看到」,不是
            # 「鎖錯了」—— 當成不一致會讓每次過場都往解鎖推一把。
            return True
        self._checks.append(matched)
        if len(self._checks) < _VERIFY_WINDOW:
            return True
        disagreed = self._checks.count(False)
        if disagreed * 2 <= len(self._checks):
            return True

        logger.warning(
            "牌桌校正 {} 與畫面對不上(最近 {} 次複查有 {} 次不一致),重新校正。"
            "常見原因是鎖定當下畫面還停在載入或選單,而不是牌桌上。",
            self._result.table_rect,
            len(self._checks),
            disagreed,
        )
        self.reset()
        return False

    def _matches_lock(self, frame: Frame) -> bool | None:
        """這一幀單獨算出來的答案,跟鎖定值算不算同一個。

        Returns:
            一致 / 不一致;這一幀**問不出答案**時回 ``None``。
        """
        assert self._result is not None
        candidate = self._calibrator.calibrate(frame)
        if candidate.source == "fallback":
            return None
        locked, now = self._result.table_rect, candidate.table_rect
        limit = min(frame.size.width, frame.size.height) * _VERIFY_TOLERANCE
        drift = max(
            abs(now.x - locked.x),
            abs(now.y - locked.y),
            abs(now.width - locked.width),
            abs(now.height - locked.height),
        )
        return drift <= limit

    def _accepts(self, candidate: Calibration) -> bool:
        """這個候選值不值得納入中位數。

        只擋兩種:整張影像的退路(``fallback``),以及寬高比離譜到「這一幀根本
        沒看到牌桌」的程度。**不以「有警告」為拒絕條件** —— 正確答案本身就會
        因為偏離 16:9 達 0.7% 而帶警告。
        """
        if candidate.source == "fallback":
            return False
        cfg = self.config
        deviation = abs(candidate.aspect - cfg.aspect_ratio) / cfg.aspect_ratio
        if deviation > cfg.aspect_reject:
            logger.debug(
                "丟棄候選 {}:寬高比偏離 {:.1%},超過 {:.1%}",
                candidate.table_rect,
                deviation,
                cfg.aspect_reject,
            )
            return False
        return True

    def _diagnose(self, candidate: Calibration) -> str | None:
        """現在該不該拒絕鎖定,以及理由。可以鎖就回 ``None``。"""
        if candidate.source == "fallback" and self.canvas.value is not None:
            return candidate.warnings[0] if candidate.warnings else "畫布尺寸對不上目前的視窗"

        if len(self._recent) < self.config.stabilize_frames:
            return None  # 樣本還太少,說什麼都太早
        rejected = self._recent.count(False)
        if rejected * 2 <= len(self._recent):
            return None
        return (
            f"最近 {len(self._recent)} 幀有 {rejected} 幀認不出牌桌邊界,"
            "多於認得出來的。這種時候取中位數只會鎖進一組系統性偏掉的答案"
            "(實測會讓手牌辨識從 96% 掉到 29%)。"
            "請在向聽分析頁選一個固定畫布尺寸,或用 manual_table_rect 手動指定。"
        )

    def _settle(self, frame: Frame) -> Calibration:
        """對蒐集到的候選取逐分量中位數,組成最終結果。"""
        xs = [r.x for r in self._samples]
        ys = [r.y for r in self._samples]
        ws = [r.width for r in self._samples]
        hs = [r.height for r in self._samples]
        rect = Rect(round(median(xs)), round(median(ys)), round(median(ws)), round(median(hs)))

        warnings: list[str] = []
        short_edge = min(frame.size.width, frame.size.height)
        limit = short_edge * _SPREAD_WARN_RATIO
        spreads = {
            "x": max(xs) - min(xs),
            "y": max(ys) - min(ys),
            "寬": max(ws) - min(ws),
            "高": max(hs) - min(hs),
        }
        noisy = {k: v for k, v in spreads.items() if v > limit}
        if noisy:
            detail = "、".join(f"{k} 相差 {v} px" for k, v in noisy.items())
            warnings.append(
                f"各幀的校正結果不一致({detail},門檻 {limit:.0f} px)。"
                f"已取中位數 {rect},但畫面可能有動畫或遮擋干擾邊界偵測。"
                "若後續 ROI 明顯偏移,請用 manual_table_rect 手動指定。"
            )

        if self._rejected > len(self._samples):
            warnings.append(
                f"丟棄的候選({self._rejected})多於採用的({len(self._samples)})。"
                "常見原因是畫面停在載入畫面或選單,而不是牌桌上。"
            )

        for message in warnings:
            logger.warning(message)

        return Calibration(
            table_rect=rect,
            image_size=frame.size,
            scale=frame.scale,
            source="auto",
            warnings=tuple(warnings),
        )
