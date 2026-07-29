"""牌面分類測試。

模板來自雀魂官方資源(``tools/fetch_tiles.py``),查詢來自真實遊戲畫面 ——
兩者是不同的來源,所以這組測試真的在驗證「貼圖能不能認出 3D 渲染的牌」,
不是在驗證「同一份資料跟自己一樣」。

``EXPECTED`` 裡的答案是目視確認過的。牌種不確定的幾張(筒子索子的點數)
**刻意不列** —— 一個標錯的期望值比沒有期望值更糟,它會讓未來真正的錯誤
看起來像通過。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from majsoul_copilot.mjai.tiles import ms_to_mjai
from majsoul_copilot.vision.tiles.classify import (
    FACE_RATIO,
    Match,
    TemplateSet,
    classify,
    classify_hand,
)
from majsoul_copilot.vision.tiles.hand import read_hand

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

#: 三張參考畫面裡**目視確認過**的牌。鍵是「素材代號 + 槽位」,
#: 素材 B 的第 13 個是摸進來的那張(白板)。
EXPECTED = {
    "hand_13_full": {0: "3m", 1: "3m", 4: "7m", 5: "7m", 6: "9m", 7: "9m",
                     8: "5p", 10: "8p", 11: "9p", 12: "9p"},
    "hand_14_with_draw": {0: "3m", 1: "8m", 2: "8m", 3: "8p", 4: "8p", 5: "1s",
                          10: "9s", 11: "1z", 12: "3z", 13: "5z"},
    "hand_4_with_melds": {0: "4z", 1: "4z", 2: "7z", 3: "7z"},
}


@pytest.fixture(scope="module")
def templates() -> TemplateSet:
    return TemplateSet.load()


def _tiles(name: str) -> tuple[np.ndarray, list]:
    roi = cv2.imread(str(FIXTURES / f"{name}.png"))
    assert roi is not None, f"讀不到 fixture {name}"
    hand = read_hand(roi)
    return roi, list(hand.concealed) + ([hand.drawn] if hand.drawn else [])


class TestTemplateSet:
    def test_all_thirty_seven_tiles_are_present(self, templates: TemplateSet) -> None:
        """34 種牌 + 3 種赤寶牌。少一種就會在實戰時靜默地認成別的牌。"""
        assert len(templates) == 37
        for suit in "mps":
            for rank in range(10):  # 0 是赤五
                assert f"{rank}{suit}" in templates
        for honor in range(1, 8):
            assert f"{honor}z" in templates

    def test_every_label_converts_to_mjai(self, templates: TemplateSet) -> None:
        """模板的命名必須是專案既有的雀魂記法,不然接不上 mjai 那層。"""
        assert {ms_to_mjai(label) for label in templates.labels}

    def test_missing_skin_says_how_to_fix_it(self) -> None:
        with pytest.raises(FileNotFoundError, match="fetch_tiles"):
            TemplateSet.load("no_such_skin")

    def test_an_empty_set_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="空"):
            TemplateSet({})

    def test_scaling_is_cached(self, templates: TemplateSet) -> None:
        """牌框大小在一場對局裡不變,重複縮放 37 張模板是純浪費。"""
        first = templates.scaled((40, 60))
        assert templates.scaled((40, 60)) is first


class TestAccuracy:
    @pytest.mark.parametrize("fixture", sorted(EXPECTED))
    def test_known_tiles_are_classified_correctly(
        self, fixture: str, templates: TemplateSet
    ) -> None:
        roi, boxes = _tiles(fixture)
        wrong = []
        for index, expected in EXPECTED[fixture].items():
            match = classify(roi[boxes[index].as_slice()], templates)
            if match.label != expected:
                wrong.append(f"槽 {index}: 期望 {expected},判成 {match}")
        assert not wrong, "\n".join(wrong)

    def test_every_known_tile_is_confident(self, templates: TemplateSet) -> None:
        """認對還不夠 —— 分數與差距也要過得了信心門檻,否則實戰會被自己擋掉。"""
        weak = []
        for fixture, answers in EXPECTED.items():
            roi, boxes = _tiles(fixture)
            for index in answers:
                match = classify(roi[boxes[index].as_slice()], templates)
                if not match.is_confident:
                    weak.append(f"{fixture} 槽 {index}: {match}")
        assert not weak, "\n".join(weak)

    def test_red_fives_do_not_swallow_normal_fives(
        self, templates: TemplateSet
    ) -> None:
        """赤五與普通五只差顏色和一個小紅點,是這套模板最薄的一組差距。

        灰階比對時差距只有 0.021~0.031,改成彩色後拉開到 0.063~0.113。
        這個測試盯住「有沒有人把比對改回灰階」。
        """
        roi, boxes = _tiles("hand_13_full")
        match = classify(roi[boxes[8].as_slice()], templates)  # 5p,次高必是 0p
        assert match.label == "5p"
        assert match.runner_up == "0p"
        assert match.margin > 0.05, f"普通五與赤五的差距太薄: {match}"

    def test_the_blank_white_dragon_is_not_confused_with_anything(
        self, templates: TemplateSet
    ) -> None:
        """白板牌面全空,是最容易跟空槽或其他淺色牌混淆的一張。"""
        roi, boxes = _tiles("hand_14_with_draw")
        match = classify(roi[boxes[13].as_slice()], templates)
        assert match.label == "5z"
        assert match.margin > 0.5, f"白板應該是壓倒性的: {match}"


class TestClassifyHand:
    def test_returns_one_match_per_tile(self, templates: TemplateSet) -> None:
        roi = cv2.imread(str(FIXTURES / "hand_14_with_draw.png"))
        hand = read_hand(roi)
        concealed, drawn = classify_hand(roi, hand, templates)
        assert len(concealed) == len(hand.concealed) == 13
        assert drawn is not None

    def test_the_drawn_tile_is_kept_separate(self, templates: TemplateSet) -> None:
        """對應 MJAI 的 tsumo —— 「手上有這張」與「這巡摸到這張」是兩件事。"""
        roi = cv2.imread(str(FIXTURES / "hand_14_with_draw.png"))
        concealed, drawn = classify_hand(roi, read_hand(roi), templates)
        assert drawn is not None
        assert drawn.label == "5z"
        assert "5z" not in [m.label for m in concealed]

    def test_no_drawn_tile_when_it_is_not_your_turn(
        self, templates: TemplateSet
    ) -> None:
        roi = cv2.imread(str(FIXTURES / "hand_13_full.png"))
        concealed, drawn = classify_hand(roi, read_hand(roi), templates)
        assert len(concealed) == 13
        assert drawn is None


class TestGuards:
    def test_a_greyscale_query_is_rejected(self, templates: TemplateSet) -> None:
        """彩色是赤五能分開的原因,傳灰階進來要當場擋掉而不是默默變差。"""
        with pytest.raises(ValueError, match="BGR"):
            classify(np.zeros((100, 70), np.uint8), templates)

    def test_a_tile_too_small_for_the_search_margin_is_rejected(
        self, templates: TemplateSet
    ) -> None:
        with pytest.raises(ValueError, match="太小"):
            classify(np.zeros((3, 3, 3), np.uint8), templates)

    def test_noise_is_reported_as_not_confident(self, templates: TemplateSet) -> None:
        """認不出來時要回報「不確定」,不是硬給一個看似合理的答案。"""
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 256, (207, 122, 3), dtype=np.uint8)
        assert not classify(noise, templates).is_confident

    def test_face_ratio_leaves_room_to_search(self) -> None:
        """比例逼近 1.0 就沒有平移餘裕,matchTemplate 會退化成單點比較。"""
        assert 0.5 < FACE_RATIO < 0.95


class TestMatch:
    def test_margin_is_the_gap_to_the_runner_up(self) -> None:
        match = Match("3m", 0.98, "8m", 0.77)
        assert match.margin == pytest.approx(0.21)
        assert match.is_confident

    def test_a_high_score_with_a_thin_margin_is_not_confident(self) -> None:
        """兩張牌都很像的時候,分數高不代表認得出來 —— 那正是赤五的情況。"""
        assert not Match("5p", 0.96, "0p", 0.95).is_confident

    def test_a_wide_margin_on_a_low_score_is_not_confident_either(self) -> None:
        assert not Match("3m", 0.30, "8m", 0.01).is_confident
