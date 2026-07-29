"""子程序引擎的協定與故障處理。

用 ``tests/fixtures/fake_bot.py`` 而不是真的 Mortal:壞掉的路徑(崩潰、逾時、
輸出雜訊)才是這一層真正的內容,而那些用真引擎難以重現。真引擎的驗證在
``test_engine_mortal.py``,需要建好 venv 與權重才會跑。
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from majsoul_copilot.engine import (
    Advice,
    EngineError,
    EngineTimeout,
    SubprocessEngine,
    SubprocessSpec,
)
from majsoul_copilot.mjai import Dahai, StartGame, Tsumo

FAKE_BOT = Path(__file__).resolve().parents[1] / "fixtures" / "fake_bot.py"


def engine(*flags: str, **spec_kwargs: object) -> SubprocessEngine:
    return SubprocessEngine(
        SubprocessSpec(
            name="fake",
            argv=[sys.executable, str(FAKE_BOT), *flags],
            startup_timeout=float(spec_kwargs.pop("startup_timeout", 10.0)),  # type: ignore[arg-type]
            react_timeout=float(spec_kwargs.pop("react_timeout", 10.0)),  # type: ignore[arg-type]
        ),
        **spec_kwargs,  # type: ignore[arg-type]
    )


TSUMO = Tsumo(actor=0, pai="3s")


class TestProtocol:
    def test_a_tsumo_comes_back_as_an_action(self) -> None:
        with engine() as bot:
            advice = bot.react(TSUMO)
        assert isinstance(advice, Advice)
        assert advice.action == Dahai(actor=0, pai="3s", tsumogiri=True)
        assert advice.engine == "fake"

    def test_other_events_come_back_as_no_action(self) -> None:
        """一進一出:不動作也要回一行,否則父程序分不出「不動作」與「還在算」。"""
        with engine() as bot:
            advice = bot.react(StartGame(id=0))
        assert advice.action is None
        assert not advice.is_action

    def test_engine_specific_meta_survives(self) -> None:
        """meta 不屬於 MJAI 協定,parse_event 會忽略它 —— 必須另外接住。

        Mortal 的 Q 值就在這裡面,是 UI 上「為什麼這樣打」的唯一素材。
        """
        with engine() as bot:
            advice = bot.react(TSUMO)
        assert advice.meta == {"seen": 1}

    def test_latency_is_measured(self) -> None:
        with engine("--reply-delay", "0.05") as bot:
            advice = bot.react(TSUMO)
        assert advice.latency_ms >= 50

    def test_stdout_noise_before_the_handshake_is_skipped(self) -> None:
        """子程序載入的第三方套件不保證只往 stderr 寫,不能因為一行雜訊就死掉。"""
        with engine("--noise") as bot:
            assert bot.react(TSUMO).is_action


class TestLifecycle:
    def test_start_is_idempotent(self) -> None:
        bot = engine()
        try:
            bot.start()
            bot.start()
            assert bot.is_running
        finally:
            bot.close()

    def test_close_is_idempotent(self) -> None:
        bot = engine()
        bot.start()
        bot.close()
        bot.close()
        assert not bot.is_running

    def test_reacting_before_start_is_an_error(self) -> None:
        with pytest.raises(EngineError, match="尚未啟動"):
            engine().react(TSUMO)

    def test_a_closed_engine_cannot_restart(self) -> None:
        bot = engine()
        bot.start()
        bot.close()
        with pytest.raises(EngineError, match="已經 close"):
            bot.start()


class TestFailures:
    def test_a_wrong_first_line_is_caught_at_startup(self) -> None:
        """握手是用來區分「還在載入權重」與「已經死了」的。

        第一行不是 hello,代表子程序把別的東西寫進了 stdout —— 協定從第一行
        就對不齊,後面每個回覆都會錯開一格。這種錯誤不擋在啟動,現場會表現成
        「引擎的建議總是慢一手」,幾乎不可能追。
        """
        with pytest.raises(EngineError, match="hello"), engine("--wrong-hello"):
            pass

    def test_a_silent_child_times_out_at_startup(self) -> None:
        """活著但不講話 —— 與崩潰不同,只能靠逾時發現。"""
        with pytest.raises(EngineTimeout, match="啟動"), engine("--no-hello", startup_timeout=0.5):
            pass

    def test_a_slow_startup_times_out(self) -> None:
        with pytest.raises(EngineTimeout, match="啟動"), engine(
            "--hello-delay", "5", startup_timeout=0.3
        ):
            pass

    def test_a_slow_reply_times_out(self) -> None:
        with engine("--reply-delay", "5", react_timeout=0.3) as bot, pytest.raises(EngineTimeout):
            bot.react(TSUMO)

    def test_a_crash_is_reported_with_the_return_code(self) -> None:
        with engine("--die-after", "0") as bot, pytest.raises(EngineError, match="returncode=3"):
            bot.react(TSUMO)

    def test_stderr_is_included_in_the_error(self) -> None:
        """崩潰時只給 returncode 等於沒說 —— 原因幾乎都在 stderr 上。"""
        bot = engine("--die-after", "0", "--stderr", "CUDA out of memory")
        with bot, pytest.raises(EngineError, match="CUDA out of memory"):
            bot.react(TSUMO)

    def test_an_error_reply_becomes_an_exception(self) -> None:
        with engine("--reply-error") as bot, pytest.raises(EngineError, match="壞掉了"):
            bot.react(TSUMO)

    def test_a_non_json_reply_is_skipped_and_then_times_out(self) -> None:
        """雜訊一律跳過,所以「只印雜訊不回答」的結果是逾時而不是格式錯誤。

        這個取捨是刻意的:第三方套件偶爾往 stdout 印東西很常見,為此中斷整場
        太脆弱。代價是這種壞法要等滿逾時才被發現 —— 跳過的內容會留在日誌裡,
        現場才追得回來。
        """
        with engine("--reply-garbage", react_timeout=0.5) as bot, pytest.raises(EngineTimeout):
            bot.react(TSUMO)

    def test_a_bad_argv_fails_to_start(self) -> None:
        bot = SubprocessEngine(SubprocessSpec(name="fake", argv=["/nonexistent/python"]))
        with pytest.raises(EngineError, match="啟動失敗"):
            bot.start()


class TestRestart:
    """引擎有狀態,重啟不等於復原 —— 新行程對這一局一無所知。"""

    def _history(self, bot: SubprocessEngine) -> Sequence[str]:
        return bot._history  # noqa: SLF001 - 重播的是它,測試要看得到

    def test_history_is_replayed_so_the_engine_catches_up(self) -> None:
        with engine() as bot:
            bot.react(StartGame(id=0))
            first = bot.react(TSUMO)
            bot.restart()
            second = bot.react(TSUMO)

        # fake_bot 的 meta.seen 是它自己數的事件數。重播了 2 個事件之後
        # 再送 1 個,新行程應該數到 3 —— 沒重播的話會是 1。
        assert first.meta == {"seen": 2}
        assert second.meta == {"seen": 3}

    def test_replay_can_be_turned_off(self) -> None:
        with engine(replay_on_restart=False) as bot:
            bot.react(StartGame(id=0))
            bot.react(TSUMO)
            bot.restart()
            after = bot.react(TSUMO)
        assert after.meta == {"seen": 1}, "關掉重播時新行程應該從零開始數"

    def test_history_keeps_growing_across_restarts(self) -> None:
        with engine() as bot:
            bot.react(StartGame(id=0))
            bot.restart()
            bot.react(TSUMO)
            assert len(self._history(bot)) == 2
