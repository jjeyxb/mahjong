"""設定資料模型 (pydantic)。

所有設定都有預設值,`config/default.yaml` 只是把預設值寫出來給人看與修改。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AppConfig",
    "CalibrationConfig",
    "CaptureConfig",
    "RiverGridConfig",
    "RoiConfig",
    "SeatRoiConfig",
    "WindowMatchConfig",
]

BackendName = Literal["auto", "macos", "windows", "mss"]

#: ROI 一律用相對 table_rect 的正規化 (x, y, width, height) 儲存,格式與
#: CalibrationConfig.manual_table_rect 一致 —— 轉成 NormRect 是使用端的事。
NormRectTuple = tuple[float, float, float, float]

#: 四邊形的四個角,同樣是相對 table_rect 的正規化座標。牌河用得到 —— 斜視角下
#: 平面矩形會投影成梯形,軸對齊矩形描述不了。見 :class:`RiverGridConfig`。
NormQuadTuple = tuple[
    tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WindowMatchConfig(_Base):
    """如何從一堆視窗中找出雀魂。

    比對方式是對視窗標題與擁有者程式名做「大小寫不敏感的子字串比對」。
    雀魂可能跑在瀏覽器分頁或 Steam 客戶端,所以預設同時涵蓋兩者。

    命中程式名的權重高於命中標題,見
    :meth:`~majsoul_copilot.capture.base.CaptureBackend.score_window`。
    """

    owner_patterns: list[str] = Field(
        default_factory=lambda: ["雀魂", "majsoul", "jantama"],
        description="比對擁有視窗的程式名稱。權重最高",
    )
    title_patterns: list[str] = Field(
        default_factory=lambda: ["雀魂", "majsoul", "maj-soul"],
        description="比對視窗標題。標題含使用者內容,容易假陽性,權重較低",
    )
    host_patterns: list[str] = Field(
        default_factory=lambda: ["chrome", "chromium", "edge", "firefox", "safari"],
        description=(
            "可能承載網頁版雀魂的瀏覽器程式名。**只有在標題也命中時才加分** ——"
            "否則任何一個瀏覽器視窗都會被誤選"
        ),
    )
    min_width: int = Field(
        640,
        ge=1,
        description="小於此寬度的視窗直接排除",
    )
    min_height: int = Field(360, ge=1, description="小於此高度的視窗直接排除")


class CaptureConfig(_Base):
    backend: BackendName = Field(
        "auto", description="auto 會依平台挑選,失敗時退回 mss"
    )
    retina: bool = Field(
        True,
        description=(
            "macOS 專用。True 取完整 backing-store 像素(Retina 下為 2 倍解析度),"
            "模板比對精度較好但每幀多約 2 ms。"
        ),
    )
    target_fps: float = Field(15.0, gt=0, le=120, description="擷取迴圈的目標幀率")
    window: WindowMatchConfig = Field(default_factory=WindowMatchConfig)


class CalibrationConfig(_Base):
    """牌桌矩形校正。

    校正的工作是從整個視窗影像中找出遊戲畫布,後續所有 ROI 都以這個矩形為基準
    做正規化。需要剝掉的外圍雜物包括:作業系統的視窗裝飾(macOS 標題列在
    Retina 下實測為 56 px)、遊戲自己補的黑邊、瀏覽器的分頁列與網址列。
    """

    border_tolerance: int = Field(
        20,
        ge=0,
        le=255,
        description="判定像素是否與該邊參考色相同的通道差異上限(0~255)",
    )
    border_uniformity: float = Field(
        0.7,
        gt=0.0,
        le=1.0,
        description=(
            "一整行/列要有這個比例以上的像素同色,才會被當成邊框剝除。"
            "用比例而非全等,是為了容忍標題列上的文字與按鈕、圓角的反鋸齒像素。"
        ),
    )
    max_peel_ratio: float = Field(
        0.4,
        gt=0.0,
        lt=1.0,
        description="每個方向最多剝除的尺寸比例。整張畫面同色時的唯一煞車",
    )
    min_content_ratio: float = Field(
        0.3,
        gt=0.0,
        le=1.0,
        description="剝除邊框後的面積至少要佔原影像的這個比例,否則視為偵測失敗",
    )

    aspect_ratio: float = Field(16 / 9, gt=0, description="雀魂畫布的設計寬高比")
    aspect_tolerance: float = Field(
        0.03, gt=0, description="偵測到的比例偏離 aspect_ratio 超過此相對誤差就示警"
    )
    enforce_aspect: bool = Field(
        False,
        description=(
            "偏離 aspect_ratio 時是否強制置中裁切成該比例。"
            "**預設關閉。** 實測 macOS Steam 版雀魂會把畫面撐滿視窗內容區"
            "(量到 1.796,而非 16:9 的 1.778),此時強制裁切反而會讓所有 ROI 偏移。"
            "只有在確認目標環境真的會補黑邊(例如某些瀏覽器全螢幕模式)時才開啟。"
        ),
    )

    manual_table_rect: tuple[float, float, float, float] | None = Field(
        None,
        description=(
            "手動指定牌桌矩形,格式為相對於視窗影像的正規化 (x, y, w, h)。"
            "設定後會略過自動偵測 —— 自動偵測失敗時的逃生門。"
        ),
    )


class SeatRoiConfig(_Base):
    """單一座位(相對自己的螢幕方位)的牌河 / 副露 ROI。

    兩個欄位都是**外框**,不是逐張切好的子格。牌河張數會隨牌局進行增加,
    快流局時甚至會疊成 2~3 列;副露的寬度更不固定 —— 吃/碰是 3 張一組,
    槓是 4 張一組(暗槓還會把頭尾兩張畫成蓋牌),硬切固定子格遇到槓就會
    溢出或被裁掉。框內實際有幾張牌、切到哪裡停,是掃描時(M3 的 vision 層)
    該解決的問題,不該在 ROI 定義階段就假設好結構。
    """

    river: NormRectTuple | None = Field(
        None,
        description="牌河外框。量測參考畫面時盡量挑接近流局的一局,"
        "外框高度才不會漏掉疊到第二、三列的情況",
    )
    melds: NormRectTuple | None = Field(
        None,
        description="副露區外框。量測參考畫面時盡量挑有槓出現的一局(槓比吃/碰寬),"
        "外框寬度才不會抓太窄",
    )


class RiverGridConfig(_Base):
    """四家牌河的網格四角。

    為什麼牌河除了 ``river`` 那個矩形之外還要這個
    ----------------------------------------------
    矩形只能框出「牌河大概在哪」,切不出「第幾張捨牌在哪一格」。牌河是 6x3 的
    網格,而斜視角讓它在畫面上變成梯形(上家與下家的還是斜的),用矩形等分會
    愈往邊緣偏愈多。四個角則剛好夠 —— 平面矩形投影後必定仍是四邊形,單應變換
    就能算出全部 18 格,還能把每格反扭回正矩形供模板比對用。

    這個東西**不能自動偵測**:試過凸包逼近與四邊包絡線加穩健迴歸,四家都對不
    準,因為參考畫面裡沒有任何一個牌河是滿的 18 格,右下角那格從沒被填過,
    等於在對不存在的資料外推。用 ``tools/grid_annotate.py`` 手動標。

    四個角依畫面上的視覺順序存:左上、右上、右下、左下。
    """

    own: NormQuadTuple | None = Field(None, description="自家牌河,畫面下方")
    kamicha: NormQuadTuple | None = Field(None, description="上家牌河,畫面左側")
    toimen: NormQuadTuple | None = Field(None, description="對面牌河,畫面上方")
    shimocha: NormQuadTuple | None = Field(None, description="下家牌河,畫面右側")


class RoiConfig(_Base):
    """牌桌上每個區域的 ROI。

    以「螢幕方位」而非「實際座位風」命名:雀魂的攝影機視角永遠讓自己在畫面
    最下方,上家/對面/下家對應的螢幕位置(左/上/右)因此每局都固定不變 ——
    只有「這個方位這局對應到哪個 actor index」會隨自己的座位改變,那個對應
    由 tracker 處理,不在這裡定義。

    每個 ROI 預設 ``None``,代表「尚未量測」。刻意不給一個看似合理的假座標——
    寧可讓用到它的程式碼在忘記設定時直接炸掉,也不要靜默地拿錯的框去跑辨識,
    那種錯誤只會在準確率報告上不明不白地變差,事後很難追查是哪個環節錯了。
    """

    own: SeatRoiConfig = Field(default_factory=SeatRoiConfig, description="自己的牌河 / 副露")
    own_hand: NormRectTuple | None = Field(
        None,
        description="自己的手牌外框,含摸進來的第 14 張。對手的手牌永遠蓋牌,"
        "沒有辨識價值,所以只有這一個欄位,不像 river/melds 四個方位都要。"
        "摸的那張會與暗手牌空一格,但不能獨立成一個 ROI —— 暗手牌長度隨副露數"
        "改變,那張牌的位置跟著在 x≈0.19~0.78 之間移動,沒有固定座標可框",
    )
    kamicha: SeatRoiConfig = Field(default_factory=SeatRoiConfig, description="上家,螢幕左側")
    toimen: SeatRoiConfig = Field(default_factory=SeatRoiConfig, description="對面,螢幕上方")
    shimocha: SeatRoiConfig = Field(default_factory=SeatRoiConfig, description="下家,螢幕右側")

    dora_indicators: NormRectTuple | None = Field(
        None,
        description="左上角那塊面板整塊,不只寶牌指示器。上半是 5 個寶牌槽"
        "(1 張起始 + 最多 4 張槓寶牌),下半約 1/3 高度是「立直棒 ×N」「本場棒 ×N」"
        "兩個計數 —— 供託與本場同樣是 tracker 需要的狀態,又緊貼在寶牌下方,"
        "拆成兩個 ROI 沒有好處",
    )
    round_info: NormRectTuple | None = Field(
        None,
        description="中央 HUD 方塊:局數、余牌數、四家點數,以及四角的座位風標記"
        "(莊家那個是紅底)。本場與供託不在這裡,在 dora_indicators 那塊面板",
    )

    river_grid: RiverGridConfig = Field(
        default_factory=RiverGridConfig,
        description="四家牌河的網格四角。上面那些 river 矩形只框出範圍,"
        "要切到「第幾張捨牌在哪一格」得靠這個",
    )


class AppConfig(_Base):
    log_level: str = Field("INFO", description="主控台日誌等級")
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    roi: RoiConfig = Field(default_factory=RoiConfig)
