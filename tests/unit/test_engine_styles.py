"""風格選單的內容與切換。

會出錯而且不會報錯的是**缺檔**:權重不隨專案散布,設定檔列的東西在別人機器
上很可能不存在。選單照列的話,使用者選了才炸,而炸的位置是 Qt 的 slot ——
整個程式當場關掉,他只是換了個選單。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mia.engine.styles import StyleChoice, StyleProfile


@pytest.fixture
def weights(tmp_path: Path) -> dict[str, Path]:
    """三份假的權重檔。內容無所謂 —— 這裡驗的是「在不在」。"""
    made = {}
    for name in ("base", "high", "low"):
        path = tmp_path / f"{name}.pth"
        path.write_bytes(b"not a real checkpoint")
        made[name] = path
    return made


class TestAvailability:
    def test_a_profile_with_a_missing_weight_is_not_offered(
        self, weights: dict[str, Path], tmp_path: Path
    ) -> None:
        """缺檔的選項不該出現在選單裡 —— 可選而選了就炸,比看不到糟。"""
        good = StyleProfile("有的", (weights["base"],))
        bad = StyleProfile("沒有的", (tmp_path / "never_downloaded.pth",))
        choice = StyleChoice((good, bad))
        assert [p.name for p in choice.profiles] == ["有的"]

    def test_a_profile_needs_every_weight_present(
        self, weights: dict[str, Path], tmp_path: Path
    ) -> None:
        """並排比較缺一份就不成立 —— 剩下那一份沒有東西可比。"""
        half = StyleProfile("比較", (weights["high"], tmp_path / "gone.pth"))
        assert not half.available
        assert half.missing == (tmp_path / "gone.pth",)

    def test_an_empty_profile_is_not_available(self) -> None:
        """一份權重都沒有的選項選了等於只剩規則式 baseline,那不是一個風格。"""
        assert not StyleProfile("空的", ()).available

    def test_no_profiles_at_all_is_a_legal_state(self) -> None:
        """設定沒寫、或權重全都不在 —— UI 把選單畫成停用,不是崩掉。"""
        choice = StyleChoice(())
        assert choice.value is None
        assert choice.name is None
        assert choice.weights == ()


class TestSelection:
    def test_the_requested_profile_wins(self, weights: dict[str, Path]) -> None:
        choice = StyleChoice(
            (
                StyleProfile("標準", (weights["base"],)),
                StyleProfile("愛鳴牌", (weights["high"],)),
            ),
            "愛鳴牌",
        )
        assert choice.name == "愛鳴牌"
        assert choice.weights == (weights["high"],)

    def test_an_unknown_name_falls_back_to_the_first_available(
        self, weights: dict[str, Path]
    ) -> None:
        """設定檔改過之後,ui_state 記的名字可能已經不存在。那不是錯誤。"""
        choice = StyleChoice((StyleProfile("標準", (weights["base"],)),), "上個版本的名字")
        assert choice.name == "標準"

    def test_selecting_the_same_profile_reports_no_change(
        self, weights: dict[str, Path]
    ) -> None:
        """回傳值決定要不要重啟引擎,而重啟要重載 130MB 權重。

        在選單上重選同一項是很常見的動作,每次都重啟等於每次都斷一巡建議。
        """
        choice = StyleChoice((StyleProfile("標準", (weights["base"],)),), "標準")
        assert not choice.set("標準")

    def test_selecting_another_profile_reports_a_change(
        self, weights: dict[str, Path]
    ) -> None:
        choice = StyleChoice(
            (
                StyleProfile("標準", (weights["base"],)),
                StyleProfile("愛鳴牌", (weights["high"],)),
            ),
            "標準",
        )
        assert choice.set("愛鳴牌")
        assert choice.weights == (weights["high"],)

    def test_selecting_an_unavailable_profile_does_nothing(
        self, weights: dict[str, Path], tmp_path: Path
    ) -> None:
        """濾掉的選項連設都設不進去 —— 不然選單以外的路徑還是能繞過檢查。"""
        choice = StyleChoice(
            (
                StyleProfile("標準", (weights["base"],)),
                StyleProfile("沒有的", (tmp_path / "gone.pth",)),
            ),
            "標準",
        )
        assert not choice.set("沒有的")
        assert choice.name == "標準"


class TestComparison:
    def test_a_profile_can_carry_several_weights(self, weights: dict[str, Path]) -> None:
        """「並排比較」就是一個帶兩份權重的選項 —— 功能 3 的整個賣點。

        順序就是優先序:ViewState.primary 取第一個有動作的,所以排前面的
        那份當 headline。
        """
        choice = StyleChoice(
            (StyleProfile("比較:鳴牌傾向", (weights["high"], weights["low"])),)
        )
        assert choice.weights == (weights["high"], weights["low"])


class TestCompareSlot:
    """比較對象是**第二個格子**,由使用者自己挑。

    原本是在設定檔裡寫死幾組(``比較:鳴牌傾向``),而那只覆蓋得到事先想到的
    那幾種組法。兩個格子的話 N 份權重自然就有 N×(N−1) 種。
    """

    @pytest.fixture
    def choice(self, weights: dict[str, Path]) -> StyleChoice:
        return StyleChoice(
            (
                StyleProfile("標準", (weights["base"],)),
                StyleProfile("愛鳴牌", (weights["high"],)),
                StyleProfile("不愛鳴牌", (weights["low"],)),
            )
        )

    def test_off_by_default(self, choice: StyleChoice) -> None:
        """預設不比較 —— 開著比較要多載一份 130MB 的權重。"""
        assert choice.compare is None
        assert len(choice.weights) == 1

    def test_picking_one_loads_both(
        self, choice: StyleChoice, weights: dict[str, Path]
    ) -> None:
        assert choice.set_compare("愛鳴牌")
        assert choice.weights == (weights["base"], weights["high"])

    def test_the_primary_stays_in_front(
        self, choice: StyleChoice, weights: dict[str, Path]
    ) -> None:
        """順序就是引擎的優先序,而 headline 取第一個有動作的。

        比較對象排前面的話,開一下比較就換人當主角 —— 使用者會以為主要風格
        被改掉了。
        """
        choice.set_compare("不愛鳴牌")
        assert choice.weights[0] == weights["base"]

    def test_none_turns_it_off(self, choice: StyleChoice) -> None:
        choice.set_compare("愛鳴牌")
        assert choice.set_compare(None)
        assert choice.compare is None

    def test_it_refuses_to_compare_with_itself(self, choice: StyleChoice) -> None:
        """同一份權重載兩次會得到兩個答案完全相同的引擎,永遠沒有分歧。

        **那是最糟的一種錯:它長得像一個結論**(「這兩個風格沒差別」)。
        """
        assert not choice.set_compare("標準")
        assert choice.compare is None

    def test_switching_the_primary_onto_the_compare_clears_it(
        self, choice: StyleChoice, weights: dict[str, Path]
    ) -> None:
        """主要風格換成目前的比較對象時,比較要關掉 —— 不然就是跟自己比。"""
        choice.set_compare("愛鳴牌")
        assert choice.set("愛鳴牌")
        assert choice.compare is None
        assert choice.weights == (weights["high"],)

    def test_an_unrelated_primary_switch_keeps_the_comparison(
        self, choice: StyleChoice, weights: dict[str, Path]
    ) -> None:
        """對照組:換成第三個風格時,比較對象留著。"""
        choice.set_compare("愛鳴牌")
        choice.set("不愛鳴牌")
        assert choice.compare_name == "愛鳴牌"
        assert choice.weights == (weights["low"], weights["high"])

    def test_an_unknown_name_means_no_comparison(self, choice: StyleChoice) -> None:
        """從 ui_state.json 讀回來的舊名字可能已經不在設定檔裡了。"""
        assert not choice.set_compare("上個月刪掉的那個")
        assert choice.compare is None

    def test_reselecting_the_same_one_reports_no_change(self, choice: StyleChoice) -> None:
        """回傳值決定要不要重啟引擎,而重選同一項是很常見的動作。"""
        assert choice.set_compare("愛鳴牌")
        assert not choice.set_compare("愛鳴牌")

    def test_a_remembered_comparison_is_restored(
        self, weights: dict[str, Path]
    ) -> None:
        choice = StyleChoice(
            (
                StyleProfile("標準", (weights["base"],)),
                StyleProfile("愛鳴牌", (weights["high"],)),
            ),
            "標準",
            "愛鳴牌",
        )
        assert choice.weights == (weights["base"], weights["high"])

    def test_a_remembered_comparison_equal_to_the_primary_is_dropped(
        self, weights: dict[str, Path]
    ) -> None:
        """設定檔的 default 改過之後,存著的比較對象可能剛好撞上它。"""
        choice = StyleChoice(
            (StyleProfile("標準", (weights["base"],)),), "標準", "標準"
        )
        assert choice.compare is None

    def test_it_says_both_names(self, choice: StyleChoice) -> None:
        """日誌寫的是這個字串 —— 只寫主要那個的話看不出比較開著沒。"""
        assert str(choice) == "標準"
        choice.set_compare("愛鳴牌")
        assert str(choice) == "標準 vs 愛鳴牌"
