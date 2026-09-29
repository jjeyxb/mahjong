"""`tools/advise.py` 怎麼判斷「引擎與真人一致」。

會算錯而且不會報錯的是**跳過**那一種:牌譜裡沒有「我不碰」這個事件,真人放過
一張牌是看不見的,只能從「他接下來不是鳴牌」反推。反推寫錯不會有任何徵兆 ——
一致率照樣印得出來,只是選擇不鳴的那份權重被系統性低估,而那正是副露率低的
那一端最常做的事。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from advise import _answer, _human_answer
from mia.mjai import Dahai, Pon

PON = Pon(actor=1, target=0, pai="N", consumed=["N", "N"])
DAHAI = Dahai(actor=1, pai="1s", tsumogiri=False)


class TestDeclining:
    def test_a_human_who_did_not_call_is_counted_as_declining_too(self) -> None:
        """真人下一手是自己摸打 → 他也放過了那張,與跳過的引擎一致。"""
        assert _answer(None) == _human_answer(DAHAI, declined=True)

    def test_a_human_who_called_disagrees_with_a_declining_engine(self) -> None:
        assert _answer(None) != _human_answer(PON, declined=True)

    def test_the_end_of_the_kyoku_still_counts_as_declining(self) -> None:
        """那一局就此結束(actual_next 回 None)—— 真人確實沒有鳴。"""
        assert _answer(None) == _human_answer(None, declined=True)


class TestActing:
    def test_the_same_action_agrees(self) -> None:
        assert _answer(PON) == _human_answer(PON, declined=False)

    def test_a_different_action_disagrees(self) -> None:
        assert _answer(PON) != _human_answer(DAHAI, declined=False)

    def test_no_human_action_is_never_counted_as_agreement(self) -> None:
        """比不出來就不算一致。``None`` 與任何答案都不相等。"""
        assert _answer(PON) != _human_answer(None, declined=False)

    def test_an_acting_engine_never_matches_the_declining_marker(self) -> None:
        """有動作的引擎不能因為真人也沒鳴就被算成一致 —— 它做的是別的事。"""
        assert _answer(PON) != _human_answer(DAHAI, declined=False)
