"""牌譜素材的副露答案。"""

from __future__ import annotations

from mia.eval.paipu import MeldKind, MeldTruth, decode_account, meld_truth
from mia.groundtruth.paipu import RecordStep
from mia.mjai.events import Ankan, Kakan, Pon, StartKyoku


def step(index: int, *events: object) -> RecordStep:
    return RecordStep(index, "", tuple(events))  # type: ignore[arg-type]


def kyoku() -> StartKyoku:
    return StartKyoku(
        bakaze="E", kyoku=1, honba=0, kyotaku=0, oya=0, dora_marker="1m",
        tehais=[["?"] * 13] * 4, scores=[25000] * 4,
    )


def test_account_decodes_from_the_link() -> None:
    """實測值:``230420-edda7633`` 的 ``_a409903155`` 是座位 0 那個人。"""
    url = "https://game.maj-soul.com/1/?paipu=230420-edda7633-6027-4eb4-b00a-30bfe7ee1ff5_a409903155"
    assert decode_account(url) == 70492146
    assert decode_account("https://game.maj-soul.com/1/?paipu=230420-x") is None


def test_source_is_relative_to_the_caller() -> None:
    """東家碰對家、碰上家、碰下家 —— 決定橫放那張在哪。"""
    steps = [
        step(0, kyoku()),
        step(1, Pon(actor=0, target=2, pai="N", consumed=["N", "N"])),
        step(2, Pon(actor=0, target=3, pai="C", consumed=["C", "C"])),
        step(3, Pon(actor=1, target=0, pai="W", consumed=["W", "W"])),
    ]
    after = meld_truth(steps)[-1]
    assert [m.source for m in after[0]] == [2, 3]
    assert after[1][0].source == 3, "座位 1 鳴座位 0 的,座位 0 是它的上家"


def test_kakan_upgrades_the_pon_in_place() -> None:
    """加槓不新增一組:疊在原本那組上面,位置不變。"""
    steps = [
        step(0, kyoku()),
        step(1, Pon(actor=0, target=2, pai="N", consumed=["N", "N"])),
        step(2, Ankan(actor=0, consumed=["P"] * 4)),
        step(3, Kakan(actor=0, pai="N", consumed=["N", "N", "N"])),
    ]
    after = meld_truth(steps)[-1][0]
    assert after == (
        MeldTruth(MeldKind.KAKAN, ("N",) * 4, 2),
        MeldTruth(MeldKind.ANKAN, ("P",) * 4, None),
    )


def test_a_new_kyoku_clears_the_melds() -> None:
    steps = [
        step(0, kyoku()),
        step(1, Pon(actor=0, target=2, pai="N", consumed=["N", "N"])),
        step(2, kyoku()),
    ]
    assert meld_truth(steps)[-1] == ((), (), (), ())
