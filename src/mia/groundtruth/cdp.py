"""用 Chrome DevTools Protocol 擷取網頁版雀魂的 WebSocket 流量。

為什麼這條路比 MITM 好
----------------------
這**不是**中間人攔截。瀏覽器自己完成 TLS 交握、自己解密,我們只是透過瀏覽器
原生的除錯介面把**已經解密好**的 frame 讀出來。因此:

* **不需要安裝任何憑證** —— 沒有在攔截 TLS,自然沒有信任鏈的問題。
* **不需要流量導向** —— 不需要 Proxifier、系統代理設定,也不需要 macOS 的
  系統延伸模組授權(那個必須使用者手動核准,無法自動化)。
* **不怕憑證固定** —— 客戶端信任誰跟我們無關。
* **不介入連線本身** —— 工具關掉最多就是沒在錄,不會像流量重導那樣可能
  在錄製中途把連線弄斷。

代價是**只能吃在瀏覽器裡跑的網頁版雀魂**,Steam 桌面版沒有這個介面可掛。
但 GT 錄製本來就是產生資料集的獨立階段,用哪個客戶端打不影響之後 AI
建議功能實際運作時要擷取的目標。

兩種連線方式
------------
* **launch**(預設)—— 由 Playwright 啟動自帶的 Chromium。最省事,什麼都不用設。
* **connect** —— 連到已經在跑、且開了 ``--remote-debugging-port`` 的瀏覽器。
  想用自己平常的瀏覽器(保留登入狀態、外掛)時用這個。

輸出格式與 mitmproxy 後端**完全相同**(見 :mod:`mia.groundtruth.dump`),
所以下游的解析與轉換不需要知道資料是從哪裡來的。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mia.calibration.canvas import Canvas
from mia.groundtruth.dump import DumpWriter
from mia.live.control import ControlFile
from mia.utils.logging import logger

if TYPE_CHECKING:
    from playwright.sync_api import BrowserContext, Page, WebSocket

__all__ = ["DEFAULT_MAJSOUL_URL", "DEFAULT_URL_PATTERNS", "CdpCapture", "CdpStats"]

#: 網頁版雀魂。與 Steam 桌面版走同一套協定。
DEFAULT_MAJSOUL_URL = "https://game.maj-soul.com/1/"

#: 只錄對局伺服器的連線,濾掉頁面上其他無關的 WebSocket。
DEFAULT_URL_PATTERNS = ("maj-soul", "majsoul", "mahjongsoul", "catmajsoul", "catmjstudio")


class PlaywrightMissingError(RuntimeError):
    """沒有安裝 Playwright 或它的瀏覽器。"""


@dataclass
class CdpStats:
    """一次擷取的統計。"""

    frames: int = 0
    text_frames: int = 0  # 雀魂協定走 binary,文字訊息不是我們要的
    sockets_seen: int = 0
    sockets_matched: int = 0
    duration: float = 0.0
    urls_ignored: set[str] = field(default_factory=set)

    def summary(self) -> str:
        lines = [
            f"錄到 {self.frames} 個 binary frame,{self.duration:.1f} 秒",
            f"WebSocket 連線 {self.sockets_seen} 條,其中 {self.sockets_matched} 條符合過濾條件",
        ]
        if self.text_frames:
            lines.append(f"略過文字 frame {self.text_frames} 個")
        if self.urls_ignored:
            lines.append("被濾掉的連線:")
            lines.extend(f"  {u}" for u in sorted(self.urls_ignored))
        return "\n".join(lines)


class CdpCapture:
    """透過 CDP 監聽瀏覽器的 WebSocket frame 並寫進錄影檔。"""

    def __init__(
        self,
        writer: DumpWriter,
        *,
        url_patterns: tuple[str, ...] | list[str] = DEFAULT_URL_PATTERNS,
    ) -> None:
        self.writer = writer
        self.url_patterns = [p.lower() for p in url_patterns if p]
        self.stats = CdpStats()
        self._flow_ids: dict[int, str] = {}

    # ------------------------------------------------------------------ 執行

    def run_launched(
        self,
        *,
        url: str = DEFAULT_MAJSOUL_URL,
        duration: float | None = None,
        headless: bool = False,
        user_data_dir: str | None = None,
        canvas: Canvas | None = None,
        control: ControlFile | None = None,
        stop_check: Callable[[], bool] | None = None,
    ) -> CdpStats:
        """啟動一個受控的 Chromium 並開啟雀魂。

        Args:
            user_data_dir: 指定的話會用持久化的設定檔目錄,登入狀態可以留到下次
                —— 否則每次都要重新登入,錄多場時很煩。
            canvas: 把視窗調到讓**頁面 viewport 剛好等於**這個尺寸。
                這是 MIA 這一端能為畫面辨識做的最有效的一件事,見
                :mod:`mia.calibration.canvas`。
            control: 主程式改設定時會寫這個檔案。**只有這個行程握著瀏覽器**,
                所以「使用者在 UI 上換了畫布尺寸」只能靠它送進來,見
                :mod:`mia.live.control`。
        """
        sync_playwright = _import_playwright()
        with sync_playwright() as playwright:
            # no_viewport=True:Playwright 預設會把分頁 viewport 鎖定在固定的
            # 1280x720,之後不管使用者怎麼拖動視窗邊框都不會變 —— 這正是
            # 「Chromium 視窗縮放時雀魂不會重新排版」那個問題的成因,不是雀魂
            # 網頁版本身的限制(直接用 Safari 之類的一般瀏覽器縮放完全正常)。
            # 加這個參數後分頁 viewport 才會跟著真正的視窗尺寸走。
            if user_data_dir:
                context = playwright.chromium.launch_persistent_context(
                    user_data_dir, headless=headless, no_viewport=True
                )
                page = context.pages[0] if context.pages else context.new_page()
            else:
                browser = playwright.chromium.launch(headless=headless)
                context = browser.new_context(no_viewport=True)
                page = context.new_page()

            self._attach_context(context)
            self._attach_page(page)

            logger.info("開啟 {}", url)
            page.goto(url, wait_until="domcontentloaded")
            if canvas is not None:
                _fit_canvas(page, canvas)
            logger.info("請在瀏覽器視窗中登入並開始對局。按 Ctrl-C 結束錄製。")

            self._wait(context, duration, stop_check, control=control, page=page)
            return self.stats

    def run_connected(
        self,
        endpoint: str,
        *,
        duration: float | None = None,
        stop_check: Callable[[], bool] | None = None,
    ) -> CdpStats:
        """連到已經在跑的瀏覽器(需以 ``--remote-debugging-port`` 啟動)。

        Args:
            endpoint: 例如 ``http://localhost:9222``。
        """
        sync_playwright = _import_playwright()
        with sync_playwright() as playwright:
            browser = playwright.chromium.connect_over_cdp(endpoint)
            if not browser.contexts:
                raise RuntimeError(f"{endpoint} 上沒有任何瀏覽器分頁")

            for context in browser.contexts:
                self._attach_context(context)
                for page in context.pages:
                    self._attach_page(page)

            logger.info(
                "已連上 {},監聽 {} 個分頁。請開啟雀魂並開始對局,按 Ctrl-C 結束。",
                endpoint,
                sum(len(c.pages) for c in browser.contexts),
            )
            self._wait(browser.contexts[0], duration, stop_check)
            return self.stats

    # ------------------------------------------------------------------ 內部

    def _attach_context(self, context: BrowserContext) -> None:
        """也要接之後才開的分頁 —— 雀魂登入流程可能會開新視窗。"""
        context.on("page", self._attach_page)

    def _attach_page(self, page: Page) -> None:
        page.on("websocket", self.attach_websocket)

    def attach_websocket(self, ws: WebSocket) -> None:
        """把 frame 監聽器掛到一條 WebSocket 上。

        通常由分頁事件自動呼叫;公開出來也讓測試能直接餵假的 WebSocket。
        """
        self.stats.sockets_seen += 1
        if not self._matches(ws.url):
            self.stats.urls_ignored.add(ws.url)
            logger.debug("略過不相符的 WebSocket: {}", ws.url)
            return

        self.stats.sockets_matched += 1
        flow = self._flow_id(ws)
        logger.info("已連上目標 WebSocket [{}]: {}", flow, ws.url)

        ws.on("framereceived", lambda payload: self._on_frame(payload, flow, from_client=False))
        ws.on("framesent", lambda payload: self._on_frame(payload, flow, from_client=True))
        ws.on("close", lambda _: logger.info("WebSocket [{}] 已關閉", flow))

    def _on_frame(self, payload: bytes | str, flow: str, *, from_client: bool) -> None:
        # Playwright 已經幫我們區分好型別:binary frame 直接就是 bytes,
        # 不像原始 CDP 還要自己 base64 解碼。
        if isinstance(payload, str):
            self.stats.text_frames += 1
            return
        self.writer.write(payload, from_client=from_client, flow=flow)
        self.stats.frames += 1
        if self.stats.frames % 200 == 0:
            logger.info("已錄 {} 個 frame", self.stats.frames)

    def _flow_id(self, ws: WebSocket) -> str:
        """同一條連線給同一個穩定識別。"""
        key = id(ws)
        if key not in self._flow_ids:
            self._flow_ids[key] = f"ws-{len(self._flow_ids):02d}"
        return self._flow_ids[key]

    def _matches(self, url: str) -> bool:
        if not self.url_patterns:
            return True
        lowered = url.lower()
        return any(p in lowered for p in self.url_patterns)

    def _wait(
        self,
        context: Any,
        duration: float | None,
        stop_check: Callable[[], bool] | None,
        *,
        control: ControlFile | None = None,
        page: Page | None = None,
    ) -> None:
        """撐住主執行緒直到時間到、使用者中斷,或瀏覽器被關掉。

        順便輪詢控制檔 —— 這個迴圈本來就每 250 ms 醒一次,搭順風車不必多開
        執行緒(而多一條執行緒去碰 Playwright 的同步 API 是不合法的)。
        """
        start = time.monotonic()
        try:
            while True:
                if duration is not None and time.monotonic() - start >= duration:
                    logger.info("已達指定時長")
                    break
                if stop_check is not None and stop_check():
                    logger.info("收到停止訊號")
                    break
                if control is not None and page is not None:
                    self._apply_control(control, page)
                try:
                    # 用 Playwright 自己的等待來推進事件迴圈,frame 事件才會送達。
                    context.pages[0].wait_for_timeout(250)
                except Exception:  # noqa: BLE001 - 視窗被關掉的表現方式有很多種
                    logger.info("瀏覽器已關閉")
                    break
        except KeyboardInterrupt:
            logger.info("使用者中斷")
        finally:
            self.stats.duration = time.monotonic() - start

    @staticmethod
    def _apply_control(control: ControlFile, page: Page) -> None:
        """套用主程式送來的新設定。**吞掉所有例外** —— 這條通道壞掉的後果
        只是視窗沒跟著調整,不該讓正在錄的那一場掛掉。"""
        command = control.poll()
        if command is None:
            return
        canvas = Canvas.parse(command.get("canvas"))
        if canvas is None:
            logger.info("收到指令但沒有可用的畫布尺寸,視窗維持原樣")
            return
        logger.info("主程式要求把畫布調成 {}", canvas.label)
        try:
            _fit_canvas(page, canvas)
        except Exception as exc:  # noqa: BLE001 - 視窗可能剛好被關掉
            logger.warning("調整視窗失敗:{}", exc)


#: 粗調最多試幾次。
#:
#: 需要不只一次是因為**工具列高度事前不知道**:視窗高度扣掉 viewport 高度
#: 才是它,而那要先開起來量。量到之後補上差額就對了,第二輪通常就收斂。
#: 留第三輪是給書籤列這種「改了視窗大小才出現/消失」的東西。
_FIT_ATTEMPTS = 3

#: 細調時每一軸往兩邊各試幾格 DIP。
#:
#: 粗調靠「差幾個 CSS 像素就把視窗改幾格 DIP」逼近,而那個假設在非整數 DPI
#: 縮放下**不成立** —— 視窗尺寸是 DIP,DIP 換算成 CSS 像素中間要過一次實體
#: 像素的四捨五入。實測 125% 螢幕:
#:
#: ===========  ==============
#: 視窗 DIP 寬   ``innerWidth``
#: ===========  ==============
#: 1613          1600  ← 命中
#: 1614          1600  ← 命中
#: 1615          1602  ← 跳 2
#: 1616          1603
#: ===========  ==============
#:
#: 差一格 DIP 可能讓 CSS 寬不動、也可能跳 2,所以粗調會在目標附近來回跳而
#: 收斂不到。這個對應關係**算不出來但試得出來**,所以細調改成逐格試。
#: 粗調之後誤差實測在 ±2 以內,兩邊各留 3 格綽綽有餘。
_FIT_NUDGE = 3

#: 改完視窗尺寸等多久再量。太短會量到還在動的中間值。
_FIT_SETTLE_MS = 200


def _fit_canvas(page: Page, canvas: Canvas) -> None:
    """把瀏覽器視窗調到讓頁面 viewport 剛好等於 ``canvas``。

    調的是 **viewport 而不是視窗**:視窗 1920x1080 扣掉工具列之後 viewport
    不是 16:9,雀魂就會自己補黑邊 —— 那樣畫布位置又變回要用猜的,而這整件事
    就是為了不要用猜的。

    調不到就只記警告不拋錯:螢幕放不下時瀏覽器會給一個它能給的尺寸,而
    「錄不到最理想的畫布」遠遠好過「整場對局沒開始」。真正的把關在
    :meth:`~mia.calibration.canvas.Canvas.table_rect` —— 對不上那邊會說。
    """
    session = page.context.new_cdp_session(page)
    window_id = session.send("Browser.getWindowForTarget")["windowId"]

    def viewport() -> tuple[int, int]:
        got = page.evaluate("() => [window.innerWidth, window.innerHeight]")
        return int(got[0]), int(got[1])

    def window_size() -> tuple[int, int]:
        bounds = session.send("Browser.getWindowBounds", {"windowId": window_id})["bounds"]
        return int(bounds["width"]), int(bounds["height"])

    def resize(width: int, height: int) -> tuple[int, int]:
        session.send(
            "Browser.setWindowBounds",
            {
                "windowId": window_id,
                # windowState 一定要一起送:視窗若是最大化或全螢幕,
                # 寬高會被直接忽略而且不會有任何錯誤。
                "bounds": {"windowState": "normal", "width": width, "height": height},
            },
        )
        page.wait_for_timeout(_FIT_SETTLE_MS)
        got = viewport()
        # 這條路只有在高 DPI 上才會出事,而出事時唯一有用的資訊就是
        # 「視窗 DIP 與量到的 CSS 像素之間實際是什麼對應」。留著。
        logger.debug("視窗 {}×{} DIP → viewport {}×{}", width, height, got[0], got[1])
        return got

    def miss(got: tuple[int, int]) -> int:
        """離目標差多少(兩軸絕對差之和)。0 就是剛好。"""
        return abs(canvas.width - got[0]) + abs(canvas.height - got[1])

    # ``Browser.getWindowBounds`` **不會**把 ``setWindowBounds`` 送進去的值原樣
    # 回報 —— 實測送 1614 進去,下一輪問回來是 1615、再問是 1617,每問一次漂一點
    # (那是視窗外框與 DWM 邊界的差,不是這裡該關心的東西)。所以起始值問一次
    # 就好,之後**全程以「我們送出去的值」為準**。
    #
    # 拿回報值當基準的話,基準會一路漂走,細調掃描的範圍就涵蓋不到正確答案 ——
    # 而那個失敗完全看不出是這個原因造成的:畫面上只會看到「調不到畫布」。
    requested = list(window_size())
    got = viewport()
    best = (miss(got), tuple(requested))

    # --- 粗調:差幾個 CSS 像素就把視窗改幾格 DIP ---
    # 平均而言 1 DIP ≈ 1 CSS px,所以這一步能很快逼到目標附近;
    # 但非整數 DPI 縮放下它到不了「剛好」,那是下面細調的事。
    for _ in range(_FIT_ATTEMPTS):
        if miss(got) == 0:
            logger.info("畫布已調整為 {}", canvas.label)
            return
        requested = [
            requested[0] + canvas.width - got[0],
            requested[1] + canvas.height - got[1],
        ]
        got = resize(requested[0], requested[1])
        if miss(got) < best[0]:
            best = (miss(got), (requested[0], requested[1]))

    # --- 細調:逐格試 ---
    # 兩軸分開掃(聯合掃是 7×7 = 49 次,太慢),每一軸都從目前最好的那組出發。
    # 兩軸之間可能經由捲軸互相影響,所以最多再跑一輪。
    for _ in range(2):
        for axis in (0, 1):
            base = list(best[1])
            for delta in range(-_FIT_NUDGE, _FIT_NUDGE + 1):
                if delta == 0:
                    continue
                trial = list(base)
                trial[axis] = base[axis] + delta
                score = miss(resize(trial[0], trial[1]))
                if score < best[0]:
                    best = (score, (trial[0], trial[1]))
                if score == 0:
                    logger.info("畫布已調整為 {}", canvas.label)
                    return

    # 掃完仍不剛好 —— 把視窗放回過程中最好的那一組再回報。
    # 不放回去的話會停在最後一次試的位置,那不保證是試過的裡面最好的。
    got = resize(best[1][0], best[1][1])
    if miss(got) == 0:
        logger.info("畫布已調整為 {}", canvas.label)
        return
    logger.warning(
        "調不到畫布 {},實際是 {}x{}(差 {} px)—— 螢幕放不下的話請改選小一點的尺寸",
        canvas.label,
        got[0],
        got[1],
        miss(got),
    )


def _import_playwright() -> Any:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise PlaywrightMissingError(
            "需要 Playwright 才能使用 CDP 擷取。\n"
            "  pip install playwright\n"
            "  playwright install chromium"
        ) from exc
    return sync_playwright
