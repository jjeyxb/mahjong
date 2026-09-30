"""loguru 日誌設定。

主程式與各個 CLI 工具啟動時呼叫一次 :func:`setup_logging` 即可。
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from mia.utils.paths import LOG_DIR

__all__ = ["logger", "setup_logging"]

#: Windows 主控台預設編碼是系統的 ANSI code page(繁體 Windows 是 cp950 / Big5),
#: 不是 UTF-8。這裡的工具會把視窗標題這種**外部來源、內容不可控**的字串直接
#: print 出去 —— 隨便一個瀏覽器分頁標題只要含簡體字或 emoji,cp950 就編不出來,
#: 整個工具當場 UnicodeEncodeError 崩潰。macOS / Linux 的預設 locale 幾乎都是
#: UTF-8,這個坑只存在於 Windows。reconfigure 是 Python 3.7+ 才有的方法,
#: 用 errors="backslashreplace" 而不是靜默丟棄,編不出來的字元至少看得出原始碼點。
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")

_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> "
    "<level>{level: <8}</level> "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
    "<level>{message}</level>"
)

_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"

_configured = False


def setup_logging(
    level: str = "INFO",
    *,
    log_dir: Path | None = LOG_DIR,
    force: bool = False,
) -> None:
    """設定全域 logger。

    Args:
        level: 主控台的最低輸出等級。
        log_dir: 檔案輸出目錄;傳 ``None`` 則不寫檔。檔案一律記錄 DEBUG 等級,
            以便事後追查辨識錯誤發生當下的完整脈絡。
        force: 預設重複呼叫不會有作用(方便函式庫程式碼安全呼叫);
            設為 True 可強制重新設定。
    """
    global _configured
    if _configured and not force:
        return

    logger.remove()
    logger.add(sys.stderr, level=level.upper(), format=_CONSOLE_FORMAT, colorize=True)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "copilot_{time:YYYYMMDD}.log",
            level="DEBUG",
            format=_FILE_FORMAT,
            rotation="20 MB",
            retention="14 days",
            encoding="utf-8",
            enqueue=True,  # 多進程/多執行緒安全,擷取執行緒會用到
        )

    _configured = True
