"""主程式 → 擷取子程序的單向控制通道。

為什麼需要它
------------
擷取子程序(``tools/gt.py cdp``)是唯一握著瀏覽器的人 —— 只有它能調視窗大小。
但那些參數是**開子程序時**用命令列傳過去的,之後主程式再改設定就送不進去了。
症狀是「在向聽分析頁換了畫布尺寸,瀏覽器沒反應」,而使用者的合理期待是它
當場就跟著變。

為什麼是檔案而不是 stdin 或 signal
----------------------------------
* **stdin**:要非阻塞地讀 pipe,POSIX 用 ``select``,而 **Windows 的 select
  只吃 socket,對 pipe 無效**。這個專案正要移植到 Windows,不想在那裡再踩一次。
* **signal**:只能送「發生了某件事」,送不了「新的尺寸是多少」。而且
  Windows 的 signal 支援非常有限。
* **檔案**:兩邊都只用 ``pathlib`` 與 ``json``,行為在三個平台上完全一樣,
  而且**留得下痕跡** —— 事後看得到主程式到底送了什麼過去。

代價是最多晚一個輪詢週期(250 ms)才生效。對「改視窗大小」這種操作,
那遠低於使用者按下去到眼睛看到的時間。

寫入是原子的
------------
先寫暫存檔再 ``os.replace``。直接覆寫的話,子程序可能剛好讀到寫了一半的
JSON —— 而那會被當成「格式壞掉」丟掉,於是這次的變更安靜地不見了。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mia.utils.logging import logger

__all__ = ["ControlFile"]


class ControlFile:
    """一個小小的 JSON 檔,主程式寫、子程序輪詢。

    Args:
        path: 檔案位置。通常放在該場錄影檔旁邊 —— 一場一個,不會跨場殘留。

    Note:
        讀寫**都不拋例外**。這條通道是「錦上添花」:壞掉的後果只是視窗沒跟著
        調整,不該讓正在錄的那一場掛掉。
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._seen: int = -1

    def write(self, **payload: Any) -> None:
        """送一份新的設定過去。每次寫都會讓序號前進。"""
        self._seen += 1
        data = {"seq": self._seen, **payload}
        tmp = self.path.with_suffix(".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            logger.debug("寫不了控制檔 {}:{}", self.path, exc)

    def poll(self) -> dict[str, Any] | None:
        """有沒有**新的**指令。沒有變化(或檔案不存在)時回 ``None``。

        靠內容裡的 ``seq`` 判斷新舊,不是靠 mtime —— 檔案系統的時間解析度在
        某些平台上只到秒,同一秒內的兩次變更會被當成沒變。
        """
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            logger.debug("讀不了控制檔 {}:{}", self.path, exc)
            return None
        if not isinstance(raw, dict):
            return None
        seq = raw.get("seq")
        if not isinstance(seq, int) or seq <= self._seen:
            return None
        self._seen = seq
        return raw
