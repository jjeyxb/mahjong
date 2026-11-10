"""發布版的封包擷取入口只開放網頁版瀏覽器擷取。

開發版的 ``tools/gt.py`` 還有自動打牌(``--autoplay``、``--cpu-room``)與兩種 MITM 模式
(``proxy``、``local``)。前者只給對電腦的房間錄素材用,後者要使用者自己裝並信任 CA 憑證,
兩者都不隨發布版提供。發布版的 ``MIA --capture ...`` 能叫到整個 ``tools/gt.py`` 命令列,
所以在這裡用白名單擋:模式只能是 ``cdp``,參數只能是一般錄製用得到的那幾個。
建置時另外把自動打牌與 mitmproxy 的模組排除(``tools/package.py``),這裡是第二道。
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["ALLOWED_MODE", "CaptureRefused", "checked_capture_args"]

#: 發布版唯一開放的擷取模式。
ALLOWED_MODE = "cdp"

#: ``cdp`` 底下允許的參數,值表示要不要帶一個值。
_CDP_OPTIONS: dict[str, bool] = {
    "--out": True,
    "--filter": True,
    "--url": True,
    "--connect": True,
    "--duration": True,
    "--headless": False,
    "--user-data-dir": True,
    "--control": True,
    "--canvas": True,
}

#: 子命令之前允許的參數。
_GLOBAL_OPTIONS: dict[str, bool] = {"--log-level": True}


class CaptureRefused(ValueError):
    """發布版不提供的擷取參數。"""


def checked_capture_args(args: Sequence[str]) -> list[str]:
    """檢查 ``MIA --capture`` 後面的參數,不在白名單裡就丟 :class:`CaptureRefused`。"""
    tokens = list(args)
    index = 0
    mode: str | None = None
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("-"):
            name, has_value = token.partition("=")[0], "=" in token
            allowed = _CDP_OPTIONS if mode is not None else _GLOBAL_OPTIONS
            if name not in allowed:
                raise CaptureRefused(f"發布版不提供這個擷取參數:{name}")
            if allowed[name] and not has_value:
                if index + 1 >= len(tokens):
                    raise CaptureRefused(f"{name} 缺少值")
                index += 1
            elif not allowed[name] and has_value:
                raise CaptureRefused(f"{name} 不接受值")
        elif mode is None:
            if token != ALLOWED_MODE:
                raise CaptureRefused(f"發布版只提供瀏覽器擷取(cdp),不提供 {token}")
            mode = token
        else:
            raise CaptureRefused(f"多餘的參數:{token}")
        index += 1
    if mode is None:
        raise CaptureRefused("缺少擷取模式")
    return tokens
