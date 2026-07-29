"""mitmproxy addon:把雀魂的 WebSocket frame 原封不動錄成 JSONL。

這是 GT 擷取的**兩種後端之一**(另一種是
:mod:`majsoul_copilot.groundtruth.cdp`,走 Chrome DevTools Protocol)。
兩者寫出的檔案格式完全相同,見 :mod:`majsoul_copilot.groundtruth.dump`。

這條路是真正的 MITM,因此需要兩件 CDP 不需要的前置作業:

1. **安裝並信任 mitmproxy 的 CA 憑證**(否則 TLS 交握會失敗)
2. **把流量導向 mitmproxy**(否則封包根本不會經過這裡,錄出來是空檔案)

好處是它**能錄 Steam 桌面版**的對局,而 CDP 只能錄瀏覽器裡的網頁版。

這個 addon 刻意**只做錄製,不做解析** —— 它跑在 mitmproxy 的事件迴圈裡,
任何例外都可能中斷代理,所以把相對容易出錯的解析移到離線階段。

啟動方式見 ``tools/gt.py``。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from majsoul_copilot.groundtruth.dump import DumpWriter

__all__ = ["ENV_OUTPUT", "ENV_URL_FILTER", "WebSocketDump", "addons"]

ENV_OUTPUT = "MAJSOUL_WS_DUMP"
ENV_URL_FILTER = "MAJSOUL_WS_FILTER"

# 雀魂的對局伺服器網域。過濾掉其他流量,免得夾雜無關的 WebSocket。
DEFAULT_URL_FILTER = "maj-soul,majsoul,mahjongsoul,catmajsoul,catmjstudio"


class WebSocketDump:
    """把符合條件的 WebSocket frame 寫成 JSONL。"""

    def __init__(self, output: Path | str | None = None, url_filter: str | None = None) -> None:
        target = output or os.environ.get(ENV_OUTPUT) or "ws_dump.jsonl"
        self.writer = DumpWriter(target)

        patterns = url_filter or os.environ.get(ENV_URL_FILTER) or DEFAULT_URL_FILTER
        self.url_patterns = [p.strip().lower() for p in patterns.split(",") if p.strip()]

        self.skipped_flows = 0
        self._ignored_urls: set[str] = set()

    @property
    def path(self) -> Path:
        return self.writer.path

    @property
    def frames(self) -> int:
        return self.writer.frames

    # --- mitmproxy hooks ---

    def running(self) -> None:
        print(f"[majsoul] WebSocket 錄製中 → {self.path}")
        print(f"[majsoul] 網址過濾: {self.url_patterns}")

    def websocket_start(self, flow: Any) -> None:
        url = flow.request.pretty_url
        if self._matches(url):
            print(f"[majsoul] 已連上目標 WebSocket: {url}")
        else:
            self.skipped_flows += 1
            print(f"[majsoul] 忽略不相符的 WebSocket: {url}")

    def websocket_message(self, flow: Any) -> None:
        if not self._matches(flow.request.pretty_url):
            return
        message = flow.websocket.messages[-1]
        if message.is_text:
            return  # 雀魂的協定走 binary frame,文字訊息不是我們要的

        self.writer.write(
            message.content, from_client=message.from_client, flow=self._flow_id(flow)
        )
        if self.frames % 200 == 0:
            print(f"[majsoul] 已錄 {self.frames} 個 frame")

    def websocket_end(self, flow: Any) -> None:  # noqa: ARG002
        print(f"[majsoul] WebSocket 結束,累計 {self.frames} 個 frame")

    def done(self) -> None:
        frames = self.frames
        self.writer.close()
        print(f"[majsoul] 錄製結束:{frames} 個 frame → {self.path}")

    # --- internals ---

    @staticmethod
    def _flow_id(flow: Any) -> str:
        """WebSocket 連線的穩定識別。

        每條連線有自己獨立的 msg_id 序列,少了這個欄位,離線解析時多條連線的
        請求/回應會互相配錯。mitmproxy 的 flow 帶有 ``.id``(UUID);
        萬一沒有就退回物件位址(同一次執行內夠穩定)。
        """
        return str(getattr(flow, "id", None) or f"obj-{id(flow):x}")

    def _matches(self, url: str) -> bool:
        if not self.url_patterns:
            return True
        lowered = url.lower()
        matched = any(p in lowered for p in self.url_patterns)
        if not matched:
            self._ignored_urls.add(url)
        return matched


# mitmproxy 以 `-s 這個檔案` 載入時會取用模組層級的 addons
addons = [WebSocketDump()]
