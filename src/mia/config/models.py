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
    "RoiConfig",
    "WindowMatchConfig",
]

BackendName = Literal["auto", "macos", "windows", "mss"]

#: ROI 一律用相對 table_rect 的正規化 (x, y, width, height) 儲存,格式與
#: CalibrationConfig.manual_table_rect 一致 —— 轉成 NormRect 是使用端的事。
NormRectTuple = tuple[float, float, float, float]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WindowMatchConfig(_Base):
    """如何從一堆視窗中找出雀魂。

    比對方式是對視窗標題與擁有者程式名做「大小寫不敏感的子字串比對」。
    雀魂可能跑在瀏覽器分頁或 Steam 客戶端,所以預設同時涵蓋兩者。

    命中程式名的權重高於命中標題,見
    :meth:`~mia.capture.base.CaptureBackend.score_window`。
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

    canvas: str | None = Field(
        None,
        description=(
            "指定遊戲畫布尺寸(邏輯像素,例如 1920x1080)。設了就**不跑邊框剝除** —— "
            "直接由影像尺寸推導 table_rect 並驗證對不對得上。剝除是啟發式,"
            "實測在 900x593 的視窗下 91% 的候選被丟棄、剩下的還是錯的"
            "(見 mia.calibration.canvas)。MIA 自己開瀏覽器時會把 viewport "
            "調到剛好這個尺寸;不是 MIA 開的就要自己確認視窗大小對得上。"
            "留空表示自動偵測"
        ),
    )

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
    aspect_reject: float = Field(
        0.10,
        gt=0.0,
        description=(
            "StableCalibrator 專用:偏離 aspect_ratio 超過此相對誤差的候選直接丟棄,"
            "不納入中位數。比 aspect_tolerance 寬鬆是刻意的 —— 那個是「示警」的門檻,"
            "這個是「這一幀根本沒看到牌桌」的門檻(實測失敗樣本落在 1.51~2.24)"
        ),
    )
    stabilize_frames: int = Field(
        15,
        ge=1,
        description=(
            "StableCalibrator 要蒐集幾個**通過檢查**的候選才鎖定結果。"
            "15 幀在 15 fps 下約 1 秒。單幀校正是啟發式,會被和了動畫、角色立繪、"
            "暗轉干擾 —— 實測 216 幀的同一場錄影產生了 21 種不同的 table_rect,"
            "其中 16% 明顯錯誤。取多幀中位數是為了不讓那 16% 被鎖進整場對局"
        ),
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


class RoiConfig(_Base):
    """牌桌上 CV 需要辨識的區域。

    **只有自己的手牌。** 牌河、副露、寶牌指示區、局數 HUD 這些狀態現在一律
    由封包(``groundtruth`` 那條路)提供,不再靠影像辨識 —— 決策過程見
    ``docs/decisions.md``。四家牌河的 ROI 與網格四角曾經量測完成,量測值保留
    在該文件中,若日後要重啟牌河 CV 可以直接取用。

    預設 ``None`` 代表「尚未量測」。刻意不給一個看似合理的假座標 —— 寧可讓
    用到它的程式碼在忘記設定時直接炸掉,也不要靜默地拿錯的框去跑辨識,那種
    錯誤只會在準確率報告上不明不白地變差,事後很難追查是哪個環節錯了。
    """

    own_hand: NormRectTuple | None = Field(
        None,
        description="自己的手牌外框,含摸進來的第 14 張。對手的手牌永遠蓋牌,"
        "沒有辨識價值。摸的那張會與暗手牌空一格,但不能獨立成一個 ROI —— "
        "暗手牌長度隨副露數改變,那張牌的位置跟著在 x≈0.19~0.78 之間移動,"
        "沒有固定座標可框",
    )


class AppConfig(_Base):
    log_level: str = Field("INFO", description="主控台日誌等級")
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    roi: RoiConfig = Field(default_factory=RoiConfig)
