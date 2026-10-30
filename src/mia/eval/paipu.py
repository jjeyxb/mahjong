"""``tools/paipu.py`` 錄的牌譜素材 → 每一張圖的副露答案。

為什麼可以逐張給答案
--------------------
工具是一次按一步、一步截一張,所以**第 k 張圖就是紀錄的第 k+1 步**
(第 0 張是配牌之後,見 ``tools/paipu.py`` 的說明)。不必像
:mod:`mia.eval.sequence` 那樣用 CV 自己的讀數去對齊。

副露答案比 :class:`~mia.mjai.table.TableTracker` 多記兩件事,都是畫面上
看得到、而且第二階段要用的(``docs/cv_mjai.md`` §3.7):

* **從誰那裡鳴的** —— 決定橫放那張在哪(上家最左、對家中間、下家最右)。
* **加槓** —— 加的那張疊在橫放那張上面,寬度不變;與大明槓長得不一樣。

局與局之間要重新對齊
--------------------
「第 k 張 = 第 k+1 步」只在**一局之內**成立。一局結束之後要多按幾次才會進到
下一局:實測流局多按兩次(流局畫面、點數結算),而紀錄裡只有一筆
``RecordNoTile``。和了要多按幾次跟役種數量有關,事前算不出來。

所以每一局開頭用畫面自己對齊:局結束之後,第一張「四家牌河都是空的而且
都讀得到」的圖就是新一局的配牌 —— 與 :class:`~mia.tracker.kyoku.KyokuWatcher`
同一個判斷。中間那幾張(結算畫面)沒有答案。

視角是哪一家
------------
牌譜連結的 ``_a`` 是視角那個人的帳號,編碼過。換視角(``--view``)之後
再依相對位置轉。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import cv2

from mia.groundtruth.paipu import RecordStep, replay_steps
from mia.groundtruth.schema import LiqiSchema
from mia.mjai.events import Ankan, Chi, Daiminkan, Kakan, Pon, StartKyoku
from mia.utils.geometry import Rect
from mia.vision.tiles.river import DEFAULT_CORNERS, SEATS, read_river

__all__ = [
    "MeldKind",
    "MeldTruth",
    "PaipuSession",
    "decode_account",
    "load_session",
    "meld_truth",
]

#: ``tools/paipu.py --view`` 的名稱 → 相對於起始視角差幾家。
VIEW_OFFSETS = {None: 0, "shimocha": 1, "toimen": 2, "kamicha": 3}

#: 新一局的空桌最多往後找幾張。實測流局是 2。
_BOUNDARY_SEARCH = 12

#: 相對位置的名稱,與 ``river.DEFAULT_CORNERS`` 的鍵一致。
POSITIONS = ("own", "shimocha", "toimen", "kamicha")


class MeldKind(Enum):
    CHI = "chi"
    PON = "pon"
    DAIMINKAN = "daiminkan"
    KAKAN = "kakan"
    ANKAN = "ankan"


@dataclass(frozen=True, slots=True)
class MeldTruth:
    """一組副露。

    Attributes:
        kind: 種類。碰之後加槓,種類會從 ``PON`` 變成 ``KAKAN``,位置不變。
        tiles: 牌,不分順序。
        source: 從誰那裡鳴的,**相對於鳴牌的人**:1 下家、2 對家、3 上家。
            暗槓是 ``None``。
    """

    kind: MeldKind
    tiles: tuple[str, ...]
    source: int | None


def decode_account(url: str) -> int | None:
    """牌譜連結的 ``_a`` → 帳號。沒有 ``_a`` 回 ``None``。

    編碼是 ``a = 1358437 + ((7 * id + 1117113) ^ 86216345)``。實測
    ``230420-edda7633`` 的 ``_a409903155`` 解出 70492146,正是座位 0 的帳號。
    """
    match = re.search(r"_a(\d+)", url)
    if match is None:
        return None
    raw = ((int(match.group(1)) - 1358437) ^ 86216345) - 1117113
    return raw // 7 if raw % 7 == 0 else None


def meld_truth(steps: list[RecordStep]) -> list[tuple[tuple[MeldTruth, ...], ...]]:
    """每一步之後,四家(依**座位**)的副露。第 i 個元素是做完第 i 步之後。"""
    melds: list[list[MeldTruth]] = [[] for _ in range(4)]
    out = []
    for step in steps:
        for event in step.events:
            if isinstance(event, StartKyoku):
                melds = [[] for _ in range(4)]
            elif isinstance(event, Chi | Pon | Daiminkan):
                kind = {Chi: MeldKind.CHI, Pon: MeldKind.PON, Daiminkan: MeldKind.DAIMINKAN}[
                    type(event)
                ]
                melds[event.actor].append(
                    MeldTruth(kind, (event.pai, *event.consumed), (event.target - event.actor) % 4)
                )
            elif isinstance(event, Ankan):
                melds[event.actor].append(MeldTruth(MeldKind.ANKAN, tuple(event.consumed), None))
            elif isinstance(event, Kakan):
                group = melds[event.actor]
                for index, meld in enumerate(group):
                    if meld.kind is MeldKind.PON and sorted(event.consumed) == sorted(meld.tiles):
                        group[index] = replace(
                            meld, kind=MeldKind.KAKAN, tiles=(*meld.tiles, event.pai)
                        )
                        break
        out.append(tuple(tuple(m) for m in melds))
    return out


@dataclass(frozen=True, slots=True)
class PaipuSession:
    """一份牌譜素材。

    Attributes:
        frames: 第 k 張圖的路徑。
        seat: 這份素材的視角是哪個座位。
        steps: 第 k 張圖是紀錄的第幾步。``None`` 是局與局之間的結算畫面,
            沒有答案。
        melds: 第 k 張圖的答案,依**相對位置**(:data:`POSITIONS`)。
            結算畫面那幾張是空的,評分時要跳過(看 :attr:`steps`)。
    """

    frames: list[Path]
    seat: int
    steps: list[int | None]
    melds: list[dict[str, tuple[MeldTruth, ...]]]


def load_session(directory: Path, schema: LiqiSchema | None = None) -> PaipuSession:
    """讀 ``tools/paipu.py`` 的輸出目錄。"""
    schema = schema or LiqiSchema.load()
    source = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    head = json.loads((directory / "head.json").read_text(encoding="utf-8"))
    seats = {a["account_id"]: a["seat"] for a in head["accounts"]}

    account = decode_account(source["url"])
    if account not in seats:
        raise ValueError(f"{directory}: 連結的視角 {account} 不在這場的玩家裡")
    seat = (seats[account] + VIEW_OFFSETS[source.get("view")]) % 4

    names = [a["nickname"] for a in sorted(head["accounts"], key=lambda a: a["seat"])]
    steps = replay_steps(schema, (directory / "record.bin").read_bytes(), seat=seat, names=names)
    after = meld_truth(steps)

    frames = sorted((directory / "frames").glob("*.png"))
    at = _frame_steps(frames, steps)
    empty: dict[str, tuple[MeldTruth, ...]] = {p: () for p in POSITIONS}
    melds = [
        {POSITIONS[(s - seat) % 4]: after[i][s] for s in range(4)} if i is not None else empty
        for i in at
    ]
    return PaipuSession(frames, seat, at, melds)


def _frame_steps(frames: list[Path], steps: list[RecordStep]) -> list[int | None]:
    """每張圖是第幾步。見模組說明「局與局之間要重新對齊」。"""
    starts = [
        s.index for s in steps if any(isinstance(e, StartKyoku) for e in s.events)
    ]
    out: list[int | None] = []
    step = 1  # 第 0 張是配牌之後
    while len(out) < len(frames):
        k = len(out)
        if step in starts[1:] and not _empty_table(frames[k]):
            # 該是新一局的配牌,畫面卻還不是空桌 —— 還在結算畫面,沒有答案。
            # 找不到空桌的話(錄到一半停了)就一直沒有答案,不要硬配。
            out.append(None)
            if out[-_BOUNDARY_SEARCH:] == [None] * _BOUNDARY_SEARCH:
                out.extend([None] * (len(frames) - len(out)))
            continue
        out.append(min(step, len(steps) - 1))
        step += 1
    return out[: len(frames)]


def _empty_table(path: Path) -> bool:
    """四家牌河都是空的而且都讀得到 —— 新一局剛配完牌。"""
    image = cv2.imread(str(path))
    if image is None:
        return False
    height, width = image.shape[:2]
    table = Rect(0, 0, width, height)
    rivers = [read_river(DEFAULT_CORNERS[seat].rectify(image, table)) for seat in SEATS]
    return all(r.count == 0 and r.is_plausible() for r in rivers)
