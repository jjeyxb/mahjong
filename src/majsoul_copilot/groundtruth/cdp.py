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

輸出格式與 mitmproxy 後端**完全相同**(見 :mod:`majsoul_copilot.groundtruth.dump`),
所以下游的解析與轉換不需要知道資料是從哪裡來的。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from majsoul_copilot.groundtruth.dump import DumpWriter
from majsoul_copilot.utils.logging import logger

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
        stop_check: Callable[[], bool] | None = None,
    ) -> CdpStats:
        """啟動一個受控的 Chromium 並開啟雀魂。

        Args:
            user_data_dir: 指定的話會用持久化的設定檔目錄,登入狀態可以留到下次
                —— 否則每次都要重新登入,錄多場時很煩。
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
            logger.info("請在瀏覽器視窗中登入並開始對局。按 Ctrl-C 結束錄製。")

            self._wait(context, duration, stop_check)
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
    ) -> None:
        """撐住主執行緒直到時間到、使用者中斷,或瀏覽器被關掉。"""
        start = time.monotonic()
        try:
            while True:
                if duration is not None and time.monotonic() - start >= duration:
                    logger.info("已達指定時長")
                    break
                if stop_check is not None and stop_check():
                    logger.info("收到停止訊號")
                    break
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
