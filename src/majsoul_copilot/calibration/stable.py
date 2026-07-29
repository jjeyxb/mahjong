"""多幀穩定化的牌桌校正。

為什麼需要這一層
----------------
:meth:`~majsoul_copilot.calibration.table.TableCalibrator.calibrate` 是**單幀**
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
"""

from __future__ import annotations

from statistics import median

from majsoul_copilot.calibration.table import Calibration, TableCalibrator
from majsoul_copilot.capture.base import Frame
from majsoul_copilot.config.models import CalibrationConfig
from majsoul_copilot.utils.geometry import Rect, Size
from majsoul_copilot.utils.logging import logger

__all__ = ["StableCalibrator"]

#: 鎖定後,若各候選在任一分量上的全距超過短邊的這個比例,就在結果裡記一筆警告。
#: 不拒絕鎖定 —— 中位數本來就是為了在雜訊中給答案,但使用者該知道畫面很不穩。
_SPREAD_WARN_RATIO = 0.02


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

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()
        self._calibrator = TableCalibrator(self.config)
        self._samples: list[Rect] = []
        self._rejected = 0
        self._result: Calibration | None = None
        self._size: Size | None = None

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

    def reset(self) -> None:
        self._samples.clear()
        self._rejected = 0
        self._result = None
        self._size = None

    # ------------------------------------------------------------------ 主流程

    def feed(self, frame: Frame) -> Calibration | None:
        """餵一幀進去。回傳已鎖定的校正,或 ``None`` 表示還在蒐集。"""
        if self._size is not None and frame.size != self._size:
            logger.info(
                "影像尺寸由 {} 變為 {},重新校正", self._size, frame.size
            )
            self.reset()

        if self._result is not None:
            return self._result

        self._size = frame.size
        candidate = self._calibrator.calibrate(frame)

        # 手動指定就沒有「不穩定」可言,直接鎖定。
        if candidate.source == "manual":
            self._result = candidate
            return self._result

        if self._accepts(candidate):
            self._samples.append(candidate.table_rect)
        else:
            self._rejected += 1

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

    def _settle(self, frame: Frame) -> Calibration:
        """對蒐集到的候選取逐分量中位數,組成最終結果。"""
        xs = [r.x for r in self._samples]
        ys = [r.y for r in self._samples]
        ws = [r.width for r in self._samples]
        hs = [r.height for r in self._samples]
        rect = Rect(
            round(median(xs)), round(median(ys)), round(median(ws)), round(median(hs))
        )

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
