"""遊戲版本:網頁版與 Steam 版的差別在**資料來源少了一條**。

這個模組沒有任何相依,所以測試也不需要 Qt、不需要執行緒。
"""

from __future__ import annotations

from mia import features, platforms
from mia.platforms import STEAM, WEB, PlatformChoice


class TestWhatEachPlatformCanDo:
    def test_the_web_version_has_everything(self) -> None:
        for key in features.NAMES:
            assert platforms.supports(WEB, key), key

    def test_steam_only_has_the_screen(self) -> None:
        """Steam 版沒有封包 —— Unity 客戶端不經過瀏覽器。

        畫面那條路兩版共通:擷取的是「一個視窗」,不在乎那是瀏覽器還是
        原生程式。
        """
        assert platforms.supports(STEAM, features.VISION)
        assert not platforms.supports(STEAM, features.ADVICE)
        assert not platforms.supports(STEAM, features.DANGER)

    def test_an_unknown_platform_is_let_through(self) -> None:
        """認不得的版本一律放行。

        這個函式是用來**擋掉確定沒有來源的組合**,不是白名單。
        ``ui_state.json`` 裡留著舊版本的名字時,放行比整個功能安靜地消失好。
        """
        assert platforms.supports("未來的某一版", features.ADVICE)

    def test_the_reason_says_why_not_just_that_it_is_off(self) -> None:
        """理由要寫給使用者看 —— 講「為什麼」而不是「不支援」。

        後者會讓人以為是 MIA 還沒做,然後去設定裡找一個不存在的開關。
        """
        why = platforms.why_unsupported(STEAM, features.ADVICE)
        assert why is not None
        assert "封包" in why

    def test_a_supported_feature_has_no_reason(self) -> None:
        assert platforms.why_unsupported(WEB, features.ADVICE) is None


class TestPlatformChoice:
    def test_it_starts_on_what_it_was_given(self) -> None:
        assert PlatformChoice(STEAM).value == STEAM

    def test_an_unknown_name_falls_back_to_web(self) -> None:
        """``ui_state.json`` 是操作留下的痕跡,舊版本留下的值不該讓程式開不起來。"""
        assert PlatformChoice("web2").value == WEB

    def test_setting_the_same_value_reports_no_change(self) -> None:
        """換版本要重啟畫面辨識,而使用者在選單上重選同一項很常見。"""
        choice = PlatformChoice(WEB)
        assert choice.set(STEAM) is True
        assert choice.set(STEAM) is False

    def test_an_unknown_name_is_refused(self) -> None:
        choice = PlatformChoice(WEB)
        assert choice.set("playstation") is False
        assert choice.value == WEB

    def test_it_answers_what_the_platform_supports(self) -> None:
        choice = PlatformChoice(STEAM)
        assert choice.supports(features.VISION)
        assert not choice.supports(features.ADVICE)
        assert choice.why_unsupported(features.ADVICE) is not None

    def test_it_has_a_name_for_people(self) -> None:
        assert str(PlatformChoice(STEAM)) == "Steam 版"
