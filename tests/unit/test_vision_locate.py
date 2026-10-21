"""在使用者框的粗略範圍裡,還原出手牌 ROI。

為什麼這個模組可以用偵測,而 :mod:`mia.calibration.letterbox` 的剝除不行,
見 :mod:`mia.vision.tiles.locate` 開頭 —— 差別在搜尋範圍與對比。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mia.vision.tiles.classify import TemplateSet, classify_hand
from mia.vision.tiles.hand import read_hand
from mia.vision.tiles.locate import find_hand_roi

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

#: 雀魂牌桌的深藍。用實際的顏色而不是純黑 —— 純黑讓對比好得不真實。
TABLE = (90, 60, 40)

#: 完整暗手牌的素材。副露那張不列在這裡,理由見
#: :func:`~mia.vision.tiles.locate.find_hand_roi` 的 warning。
FULL_HANDS = ("hand_13_full", "hand_14_with_draw")


def _load(name: str) -> np.ndarray:
    image = cv2.imread(str(FIXTURES / f"{name}.png"))
    assert image is not None, name
    return image


def _padded(image: np.ndarray, pad: int) -> np.ndarray:
    """把手牌 ROI 墊進一塊桌面裡 —— 模擬使用者框大了。"""
    return cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=TABLE)


def _read(roi: np.ndarray) -> tuple[list[str], str | None, float]:
    hand = read_hand(roi)
    assert hand.is_plausible, f"讀到 {len(hand)} 張,不是合法手牌"
    concealed, drawn = classify_hand(roi, hand, TemplateSet.load(), headroom=0)
    matches = [*concealed, *([drawn] if drawn else [])]
    label = drawn.label if drawn else None
    return [m.label for m in concealed], label, float(np.mean([m.score for m in matches]))


class TestFindingTheRoi:
    @pytest.mark.parametrize("name", FULL_HANDS)
    @pytest.mark.parametrize("pad", [10, 40, 90])
    def test_it_reads_the_same_hand_as_the_real_roi(self, name: str, pad: int) -> None:
        """**框多鬆都認出同一手牌** —— 那正是這個模組存在的理由。

        使用者是在縮小的預覽圖上拖框的:預覽縮到 800px 寬時一個預覽像素將近
        四個影像像素,手抖幾像素就會讓每個槽位跟著歪。實測框得比真值鬆 30px
        就整手變成 ``5z 5z 5z …`` —— 13 張、合法、而且全錯。
        """
        original = _load(name)
        expected = _read(original)

        padded = _padded(original, pad)
        roi = find_hand_roi(padded)
        assert roi is not None
        crop = padded[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
        concealed, drawn, score = _read(crop)
        assert (concealed, drawn) == expected[:2]
        assert score > 0.8, f"平均比對分數只有 {score:.3f}"

    @pytest.mark.parametrize("name", FULL_HANDS)
    def test_the_slack_does_not_change_the_answer(self, name: str) -> None:
        """不同鬆緊度要給出**同一個**框,不只是「都差不多對」。"""
        original = _load(name)
        answers = {
            (r.x - pad, r.y - pad, r.width, r.height)
            for pad in (10, 40, 90)
            if (r := find_hand_roi(_padded(original, pad))) is not None
        }
        assert len(answers) == 1, f"不同鬆緊度給出不同的框:{answers}"

    def test_the_width_does_not_follow_the_tile_count(self) -> None:
        """**亮帶不等於 ROI。**

        ``hand_13_full`` 沒有摸牌,右邊空著一格,所以看得到的牌只佔 1651px,
        而 ROI 是 1813px —— 那一格也算 ROI,因為 ``HandLayout`` 的槽位是
        ROI 寬度的比例。拿亮帶當 ROI 的話,牌一少 ROI 就跟著縮,槽位全歪。
        """
        thirteen = find_hand_roi(_padded(_load("hand_13_full"), 40))
        fourteen = find_hand_roi(_padded(_load("hand_14_with_draw"), 40))
        assert thirteen is not None and fourteen is not None
        assert abs(thirteen.width - fourteen.width) < 30, (
            f"差了一張摸牌就讓 ROI 寬度從 {thirteen.width} 變成 {fourteen.width}"
        )

    def test_a_bright_distractor_above_is_ignored(self) -> None:
        """框大了會帶到玩家名牌與分數字樣,那些也是亮的。

        所以列的方向取**最長的連續亮段**,不是「第一個到最後一個亮的列」——
        後者會把名牌到手牌之間的整段桌面都算進來。
        """
        padded = _padded(_load("hand_13_full"), 80)
        padded[8:24, 200:600] = (230, 230, 230)
        roi = find_hand_roi(padded)
        assert roi is not None
        assert abs(roi.y - 80) <= 20, "被上面那條亮的拉走了"

    def test_an_empty_table_finds_nothing(self) -> None:
        """框到一塊空桌面 —— 回 None 讓呼叫端說「這裡沒有手牌」。

        回一個雜訊切出來的框會一路傳到辨識,然後整手認成白板:
        13 張、合法、全錯。
        """
        assert find_hand_roi(np.full((300, 1200, 3), TABLE, dtype=np.uint8)) is None

    def test_a_noisy_empty_table_still_finds_nothing(self) -> None:
        rng = np.random.default_rng(0)
        table = np.full((300, 1200, 3), TABLE, dtype=np.uint8)
        noise = rng.integers(-4, 5, table.shape, dtype=np.int16)
        assert find_hand_roi(np.clip(table + noise, 0, 255).astype(np.uint8)) is None

    def test_it_refuses_a_non_colour_image(self) -> None:
        assert find_hand_roi(np.zeros((10, 10), dtype=np.uint8)) is None
        assert find_hand_roi(np.zeros((0, 0, 3), dtype=np.uint8)) is None


class TestMeldsAtCalibrationTime:
    def test_a_melded_hand_does_not_produce_a_usable_roi(self) -> None:
        """**框選時不能有副露** —— 這一條釘住那個已知的限制。

        副露貼得比暗手牌緊,又隔著一大段空桌面,兩群不同牌距讓自相關抓不到
        單一週期。這只影響框選那一刻:框好之後 ROI 固定,副露照樣認得出來。

        這條測試在這裡不是因為行為「對」,而是因為它**可預期** —— 呼叫端
        據此知道一定要當場認一次給使用者看,不能把框直接存下去。
        """
        padded = _padded(_load("hand_4_with_melds"), 40)
        roi = find_hand_roi(padded)
        assert roi is not None, "它仍然會給一個框 —— 所以呼叫端必須自己驗"
        crop = padded[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
        assert not read_hand(crop).is_plausible, (
            "有副露時竟然讀出合法手牌 —— 那代表這個限制已經不存在,"
            "該回頭把 FULL_HANDS 補上這張素材"
        )
