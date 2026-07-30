"""Mortal 引擎的接線:把設定變成一個可執行的子程序。

這個模組**不含任何推論邏輯**,也不 import torch 或 libriichi —— 那些都在
子程序那一端(``engines/mortal/bot.py``,跑 Python 3.12)。這裡只負責找到
直譯器、組出命令列、把路徑檢查清楚後交給
:class:`~mia.engine.subprocess_engine.SubprocessEngine`。

為什麼不直接跑 Mortal 自己的 ``mortal.py``
------------------------------------------
Mortal 附的 ``mortal.py`` 已經是一個 MJAI stdio 迴圈,乍看可以直接用。但它
**只在有動作時輸出**,父程序無法區分「不需要動作」與「還在算」;它的
``MORTAL_REVIEW_MODE`` 雖然會補 ``none``,卻在 stdin 關閉後接著跑 GRP 分析,
而本專案還沒有 GRP 權重(見 ``docs/decisions.md``),每次關閉都會以例外收場。

所以子程序端是自己寫的一層薄殼,只做三件 ``mortal.py`` 沒做的事:握手、
補 ``none``、把例外轉成一行 JSON。推論本身仍然完全交給上游的
``MortalEngine`` 與 ``libriichi.mjai.Bot``。
"""

from __future__ import annotations

from pathlib import Path

from mia.engine.base import EngineError
from mia.engine.subprocess_engine import SubprocessEngine, SubprocessSpec
from mia.utils.paths import PROJECT_ROOT

__all__ = ["MORTAL_DIR", "mortal_engine"]

#: 子程序那一端的所有東西都在這裡:自己的 venv、clone 下來的 Mortal、bot.py
MORTAL_DIR = "engines/mortal"

_BOT = "bot.py"
_VENV_PYTHON = ".venv/bin/python"
_VENV_PYTHON_WINDOWS = ".venv/Scripts/python.exe"
_UPSTREAM = "Mortal/mortal"


def mortal_engine(
    weights: Path | str,
    *,
    seat: int = 0,
    name: str = "mortal",
    root: Path | None = None,
    react_timeout: float = 5.0,
) -> SubprocessEngine:
    """建一個還沒啟動的 Mortal 引擎。

    Args:
        weights: 權重 ``.pth`` 的路徑。
        seat: ``start_game`` 到達之前的座位。libriichi 的 ``Bot`` 在建構時就要
            知道座位,所以子程序啟動時得先給一個值;但 ``bot.py`` 收到
            ``start_game`` 會照它的 ``id`` 重建 ``Bot``(權重不重載),
            **實際座位由事件流決定**。即時模式因此可以在還不知道自己坐哪的
            時候就先把引擎起好 —— 載權重要 10 秒,不能等到發牌才開始載。
        name: 引擎名稱。同時跑多份權重比較時,這是唯一分得出誰是誰的東西。
        root: 專案根目錄,預設自動推導。測試會覆寫它。
        react_timeout: 每一手的逾時秒數。

    Returns:
        尚未啟動的引擎。呼叫 ``start()`` 或用 ``with`` 才會真的開行程。

    Raises:
        EngineError: 直譯器、``bot.py``、上游原始碼或權重任一個不存在。

    Note:
        缺什麼在這裡就查清楚,而不是等子程序自己 ImportError —— 那個錯誤
        會從 stderr 繞一大圈回來,訊息還是英文的 traceback,很難一眼看出
        是「忘了建 venv」還是「忘了下載權重」。
    """
    base_root = root or PROJECT_ROOT
    base = base_root / MORTAL_DIR
    weights_path = Path(weights).expanduser()
    if not weights_path.is_absolute():
        weights_path = base_root / weights_path

    python = base / _VENV_PYTHON
    if not python.exists():
        python = base / _VENV_PYTHON_WINDOWS

    _require(
        python,
        f"找不到子程序的 Python。依 {MORTAL_DIR}/requirements.txt 的步驟建立 3.12 venv",
    )
    _require(base / _BOT, "子程序端的腳本不見了,應該隨專案版控")
    _require(
        base / _UPSTREAM / "libriichi.so",
        "libriichi 尚未編譯。見 requirements.txt 的建置步驟(macOS 記得把 .dylib 改名成 .so)",
    )
    _require(weights_path, "權重不隨專案散布,請依 Mortal 專案的規範自行取得")

    spec = SubprocessSpec(
        name=name,
        argv=[str(python), str(base / _BOT), "--seat", str(seat), "--weights", str(weights_path)],
        # 不指定 cwd:bot.py 自己把上游的 mortal/ 插進 sys.path,從哪裡啟動都一樣
        react_timeout=react_timeout,
    )
    return SubprocessEngine(spec)


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        raise EngineError(f"找不到 {path} —— {hint}")
