"""UI 自己記得住的東西:使用者把 Overlay 拖到哪、勾了哪些選項。

為什麼不放進 config/
--------------------
``config/default.local.yaml`` 是「這台機器的設定覆寫」,而
:func:`~mia.config.loader.save_config` 寫回去的是**整份 AppConfig 的快照**。
把視窗位置存進去等於順手把當下所有預設值一起寫死 —— 日後 ``default.yaml``
改了預設,這台機器不會跟著變,而那種失效沒有任何症狀,只會讓人以為改的地方
沒生效。

再說視窗位置本來也不是設定:那是操作留下的痕跡,跟日誌與錄影同一類。
所以放在 ``data/``(本來就 gitignore,每台機器各有一份)。

壞掉了就當作沒有
----------------
這個檔案裡沒有任何「弄丟了會怎樣」的東西 —— 最壞的情況是 Overlay 回到預設
位置。所以讀取一律不拋例外:格式壞了、型別不對、檔案被手改壞,通通當成沒存過。
為了一個記住的座標讓整個 UI 開不起來是不划算的交易。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from mia.utils.logging import logger
from mia.utils.paths import DATA_DIR

__all__ = ["STATE_PATH", "UiState"]

#: 預設的存放位置。``data/`` 已在 gitignore 裡。
STATE_PATH: Path = DATA_DIR / "ui_state.json"


@dataclass
class UiState:
    """跨次啟動記住的 UI 狀態。

    Attributes:
        overlay_visible: 上次關掉程式時 Overlay 是開著的嗎。
        overlay_expanded: Overlay 要不要展開候選 Q 值。
        overlay_locked: Overlay 是否鎖定位置(鎖定 = 滑鼠穿透)。
        overlay_pos: Overlay 左上角的螢幕座標。``None`` 表示還沒拖過。
        canvas: 選定的遊戲畫布尺寸(例如 ``1920x1080``)。``None`` 為自動偵測。
    """

    canvas: str | None = None
    overlay_visible: bool = False
    overlay_expanded: bool = False
    #: 預設**不鎖定** —— 第一次開起來一定要先拖到想要的位置,而鎖定的視窗
    #: 拖不動。鎖上了才點得到底下的遊戲,那是使用者自己該做的下一個動作。
    overlay_locked: bool = False
    overlay_pos: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        #: 從哪裡讀來的就存回哪裡 —— 呼叫端只需要在 :meth:`load` 指定一次路徑。
        #: 刻意**不是** dataclass 欄位:它是「這份狀態住在哪」,不是狀態本身,
        #: 寫進 JSON 只會在檔案搬家之後變成一個對不上的舊路徑。
        self.path: Path = STATE_PATH

    @classmethod
    def load(cls, path: Path | str | None = None) -> UiState:
        """讀回上次的狀態。讀不到、或內容不合用時回一份預設值。

        之後的 :meth:`save` 會寫回**同一個**檔案,不必再指定一次。
        """
        target = Path(path) if path is not None else STATE_PATH
        state = cls()
        state.path = target
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return state
        except (OSError, ValueError) as exc:
            logger.debug("讀不了 {},用預設的 UI 狀態:{}", target, exc)
            return state
        if not isinstance(raw, dict):
            return state
        loaded = cls(**_known(raw))
        loaded.path = target
        return loaded

    def save(self, path: Path | str | None = None) -> None:
        """寫回磁碟。寫不成功只記一句日誌,不打斷使用者正在做的事。

        先寫暫存檔再 rename:直接覆寫的話,寫到一半被中斷會留下一個半截的
        JSON,下次啟動就讀不回來了。rename 在同一個檔案系統上是原子的。
        """
        target = Path(path) if path is not None else self.path
        tmp = target.with_suffix(".json.tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, target)
        except OSError as exc:
            logger.debug("存不了 {}:{}", target, exc)


def _known(raw: dict[str, Any]) -> dict[str, Any]:
    """只留下型別對得上的欄位。

    多出來的鍵直接丟掉(舊版本留下的),型別不對的也丟掉 —— 換成預設值比
    把一個字串當座標傳給 ``move()`` 好。
    """
    out: dict[str, Any] = {}
    for field in fields(UiState):
        if field.name not in raw:
            continue
        value = raw[field.name]
        if field.name == "canvas":
            # 型別對就收下,內容看不懂是 Canvas.parse 的事(它會退回自動偵測)
            if value is None or isinstance(value, str):
                out[field.name] = value
        elif field.name == "overlay_pos":
            if (
                isinstance(value, list | tuple)
                and len(value) == 2
                and all(isinstance(v, int) for v in value)
            ):
                out[field.name] = (int(value[0]), int(value[1]))
        elif isinstance(value, bool):
            out[field.name] = value
    return out
