"""介面語言。

原文就是程式裡的繁體中文,翻譯檔(``assets/i18n/<語言>.json``)以原文為鍵。找不到譯文就顯示原文,
所以漏翻的地方是中文,不會變成空白或鍵名。

兩種用法
--------
* :func:`tr`:介面自己的字。``tr("視窗置頂")``;要帶值的寫成模板,``tr("擷取失敗 {n} 次", n=3)``,
  翻譯檔的鍵是模板本身。
* :func:`translate`:別處送來、已經組好的句子,例如工作執行緒的狀態列訊息
  (``f"擷取失敗 {n} 次"`` 組出來的「擷取失敗 3 次」)。先找完全相同的原文,再拿帶 ``{名稱}`` 的
  模板去比對,比得上就把值填進譯文。那些訊息散在各個執行緒裡,在顯示的地方翻一次就好,
  不必每一句都改寫。

語言什麼時候定
--------------
第一次用到時讀介面設定(``ui_state.json`` 的 ``language``),之後整個程序不變。環境變數
``MIA_LANGUAGE`` 優先,測試用它固定成原文。模組載入時就組好的
常數(功能名稱、頁名)也會用到它,所以不能等 UI 建好才設定。換語言要重新啟動才生效。
"""

from __future__ import annotations

import json
import os
import re
import threading
from functools import lru_cache
from pathlib import Path

from mia.utils.paths import ASSETS_DIR, DATA_DIR

__all__ = ["DEFAULT_LANGUAGE", "LANGUAGES", "language", "set_language", "tr", "translate"]

#: 預設語言,也是原文的語言。
DEFAULT_LANGUAGE = "zh-TW"

#: 可選的語言。值是那個語言自己的名字,選單裡照原樣顯示,不跟著目前的語言翻。
LANGUAGES: dict[str, str] = {"zh-TW": "繁體中文", "en": "English"}

#: 翻譯檔放在哪。
CATALOG_DIR: Path = ASSETS_DIR / "i18n"

#: 介面設定檔。與 :data:`mia.ui.state.STATE_PATH` 同一個;這裡不 import 它,避免循環。
_STATE_PATH: Path = DATA_DIR / "ui_state.json"

#: 模板裡的 ``{名稱}`` 或 ``{名稱:格式}``。
_FIELD = re.compile(r"\{(\w+)(?:![rsa])?(?::[^{}]*)?\}")

_lock = threading.Lock()
_language: str | None = None
_catalog: dict[str, str] = {}
_patterns: list[tuple[re.Pattern[str], str, tuple[str, ...]]] = []


def _saved_language() -> str:
    forced = os.environ.get("MIA_LANGUAGE")
    if forced in LANGUAGES:
        return forced
    try:
        raw = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DEFAULT_LANGUAGE
    value = raw.get("language") if isinstance(raw, dict) else None
    return value if value in LANGUAGES else DEFAULT_LANGUAGE


def _load(code: str) -> dict[str, str]:
    if code == DEFAULT_LANGUAGE:
        return {}
    try:
        raw = json.loads((CATALOG_DIR / f"{code}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str) and v}


def _compile(catalog: dict[str, str]) -> list[tuple[re.Pattern[str], str, tuple[str, ...]]]:
    """帶 ``{名稱}`` 的鍵做成正規表示式。長的先比,短的模板才不會先吃掉長句的一部分。"""
    out = []
    for source, target in sorted(catalog.items(), key=lambda item: -len(item[0])):
        names = tuple(match.group(1) for match in _FIELD.finditer(source))
        if not names or len(set(names)) != len(names):
            continue
        parts = _FIELD.split(source)
        # split 的結果是 [字, 名稱, 字, 名稱, ..., 字]
        pattern = "".join(
            re.escape(part) if index % 2 == 0 else f"(?P<{part}>.+?)"
            for index, part in enumerate(parts)
        )
        out.append((re.compile(f"^{pattern}$", re.DOTALL), target, names))
    return out


def set_language(code: str) -> None:
    """換這個程序的語言。認不得的退回預設。測試與啟動時用;執行中換只影響之後組出來的字。"""
    global _language, _catalog, _patterns
    code = code if code in LANGUAGES else DEFAULT_LANGUAGE
    with _lock:
        _language = code
        _catalog = _load(code)
        _patterns = _compile(_catalog)
    _translated.cache_clear()


def language() -> str:
    """目前的語言。第一次呼叫時讀介面設定。"""
    if _language is None:
        set_language(_saved_language())
    assert _language is not None
    return _language


def tr(text: str, /, **values: object) -> str:
    """介面上的字。``values`` 填進模板(譯文與原文用同一組名稱)。"""
    language()
    target = _catalog.get(text, text)
    if not values:
        return target
    try:
        return target.format(**values)
    except (KeyError, IndexError, ValueError):
        # 譯文的欄位寫錯了:退回原文,不讓介面因為翻譯檔壞掉而出錯。
        return text.format(**values)


def translate(message: str) -> str:
    """別處組好的句子。見模組說明。"""
    if not message or language() == DEFAULT_LANGUAGE:
        return message
    return _translated(message)


@lru_cache(maxsize=4096)
def _translated(message: str, depth: int = 0) -> str:
    exact = _catalog.get(message)
    if exact is not None:
        return exact
    if depth > 1:
        return message
    for pattern, target, names in _patterns:
        match = pattern.match(message)
        if match is None:
            continue
        values = {name: _translated(match.group(name), depth + 1) for name in names}
        # 值已經是字串了,模板裡的格式(:.1f)不能再套一次。
        plain = _FIELD.sub(lambda m: "{" + m.group(1) + "}", target)
        try:
            return plain.format(**values)
        except (KeyError, IndexError, ValueError):
            return message
    return message
