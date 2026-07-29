"""真正的 Mortal 子程序。

需要建好 ``engines/mortal/.venv``、編好 libriichi、下載權重才會跑,缺任何一項
就整檔跳過 —— 那些東西不隨專案散布(權重 130 MB、libriichi 要自行編譯)。

其餘的協定與故障處理用假的子程序測,見 ``test_engine_subprocess.py``。這裡只
問一件假引擎答不出來的事:**接上真的 Mortal,它會不會動。**
"""

from __future__ import annotations

import pytest

from majsoul_copilot.engine.base import EngineError
from majsoul_copilot.engine.mortal import mortal_engine
from majsoul_copilot.mjai import Dahai, Reach, StartGame, StartKyoku, Tsumo
from majsoul_copilot.utils.paths import MODELS_DIR

WEIGHTS = MODELS_DIR / "mortal_298k.pth"

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def engine():
    try:
        bot = mortal_engine(WEIGHTS, seat=0)
    except EngineError as exc:
        pytest.skip(f"Mortal 子程序環境未就緒 — {exc}")
    with bot:
        yield bot


# 123m 456m 789m 123p + 5p 單騎聽牌
TENPAI = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"]


def test_it_starts_and_reaches_a_decision(engine) -> None:
    """開局 → 聽牌 → 摸一張無關牌,Mortal 應該有話要說。"""
    assert engine.react(StartGame(id=0)).action is None
    kyoku = StartKyoku("E", 1, 0, 0, 0, "2s", [TENPAI, *[["?"] * 13] * 3], [25000] * 4)
    assert engine.react(kyoku).action is None

    advice = engine.react(Tsumo(actor=0, pai="9s"))
    assert advice.is_action, "聽牌摸牌之後不該是「不動作」"
    assert advice.engine == "mortal"


def test_it_reports_q_values(engine) -> None:
    """meta 是 UI 上「為什麼這樣打」的唯一素材。

    子程序刻意關掉 ``enable_quick_eval`` 就是為了每一手都有 Q 值 —— 開著的話
    「明顯只有一個選擇」的那些手會跳過網路推論,meta 就是空的。
    """
    advice = engine.react(Reach(actor=0))
    assert advice.meta is not None
    assert "q_values" in advice.meta


def test_it_keeps_state_across_events(engine) -> None:
    """引擎自己從事件流重建局面 —— 立直宣言之後接得上宣言牌。"""
    advice = engine.react(Dahai(actor=0, pai="9s", tsumogiri=True))
    assert advice.action is None, "自己打完牌之後輪到別人"


def test_a_missing_weight_file_fails_early(tmp_path) -> None:
    """缺什麼在父程序就查清楚,不要等子程序自己 ImportError。

    那個錯誤會從 stderr 繞一圈回來、還是英文 traceback,很難一眼看出是
    「忘了建 venv」還是「忘了下載權重」。
    """
    with pytest.raises(EngineError, match="權重"):
        mortal_engine(tmp_path / "nope.pth", seat=0)
