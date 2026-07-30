"""AI 引擎層:把 MJAI 事件流變成打牌建議。

三種引擎,同一個介面(:class:`~mia.engine.base.AIEngine`):

* :class:`~mia.engine.dummy.DummyEngine` —— 規則式 baseline,
  行程內執行,只看向聽與進張。
* :func:`~mia.engine.mortal.mortal_engine` —— Mortal,跑在
  Python 3.12 的子程序裡(libriichi 與 torch 都裝不進主環境)。
* 日後微調出來的權重 —— 與上一項是同一個函式,只是換一個 ``.pth``。

:class:`~mia.engine.multiplex.EngineGroup` 把同一條事件流餵給
多個引擎,並排它們的答案 —— 這是功能 3「風格比較」的骨架。

``mortal_engine`` 刻意**不**在這裡匯出:它會去檢查子程序的 venv 與權重存不存在,
匯入這個套件不該有那種副作用。要用的人從
``mia.engine.mortal`` 直接 import。
"""

from mia.engine.base import Advice, AIEngine, EngineError, EngineTimeout
from mia.engine.dummy import DummyEngine
from mia.engine.multiplex import EngineGroup, GroupResult
from mia.engine.subprocess_engine import SubprocessEngine, SubprocessSpec

__all__ = [
    "AIEngine",
    "Advice",
    "DummyEngine",
    "EngineError",
    "EngineGroup",
    "EngineTimeout",
    "GroupResult",
    "SubprocessEngine",
    "SubprocessSpec",
]
