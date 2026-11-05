"""別家的手牌形狀 → 剛打的那張是摸切還是手切。

量法見 :mod:`mia.vision.tiles.backs`。這裡只記每家最近幾秒的形狀,打牌排進事件流的那一刻
問一次,之後不再改。

為什麼不等看清楚再改
--------------------
手切的樣子從打牌開始維持約一秒,而別家的牌在封包之後約 0.6 秒就排進事件流了。等到一秒再
改,事件流就得改寫,引擎要把這一局從頭重送一次(實測 0.1 ~ 0.2 秒),別家一半的牌是手切,
等於每一兩秒重送一次。所以在排進去的那一刻就決定:

* 上家:那一刻就有 94%,而上家打完就輪到自己,不能讓它等。
* 對家、下家:還沒看到手切的樣子時多等一輪(0.15 秒),92%、90%,不等的話是 86%、81%。
  它們晚一點排進去只影響後面的人,不影響自己摸牌後的建議。

十二場、別家 5050 張自己決定的捨牌(鳴牌後一定手切、立直後一定摸切,不算在內)。之前一律
填手切,只對 41%。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from mia.vision.tiles.backs import BackShape

__all__ = ["HOLD", "TsumogiriWatcher"]

#: 對家、下家還沒看到手切的樣子時,捨牌最多晚幾秒排進去。
HOLD = 0.15

#: 留幾秒的形狀。手牌原本有多長要從這裡面找,打牌那一刻的形狀是缺了牌的。
_KEEP = 3.0

#: 打牌前後多久之內的形狀算這一手的。比同一家兩手之間的間隔短得多。
_RECENT = 1.5

#: 手牌本體比原本少多少(沿線的點數)才算少了一張:對家 10、上家 6。
_SHORTER = {2: 10, 3: 6}

#: 對家要看到幾次:角色的手偶爾也切出同樣的形狀。
_NEED = {1: 1, 2: 2, 3: 1}

#: 下家分成兩塊時,總面積至少是原本的多少:被手蓋住的話會少一大截。
_WHOLE = 0.9

#: 手牌本體比這個小就是沒量到手牌(版面不同、視窗被蓋住),不下判斷。沿線量的單位是點數,
#: 數區塊的是面積。
_TOO_SMALL = {1: 5000.0, 2: 50.0, 3: 50.0}


@dataclass
class TsumogiriWatcher:
    """記三家的手牌形狀。座位依視角,1 是下家。"""

    _seen: dict[int, deque[tuple[float, BackShape]]] = field(
        default_factory=lambda: {position: deque() for position in (1, 2, 3)}
    )
    _drawn_seen: set[int] = field(default_factory=set)

    def reset(self) -> None:
        """新的一局。"""
        for history in self._seen.values():
            history.clear()
        self._drawn_seen.clear()

    def forget(self, position: int) -> None:
        """這家鳴牌了。手牌短了一截,之前看到的長度會讓之後每一手都像少了一張。"""
        self._seen[position].clear()
        self._drawn_seen.discard(position)

    def observe(self, position: int, now: float, shape: BackShape) -> None:
        history = self._seen[position]
        history.append((now, shape))
        if shape.drawn and shape.clean:
            self._drawn_seen.add(position)
        while history and history[0][0] < now - _KEEP:
            history.popleft()

    def tegiri_seen(self, position: int, now: float) -> bool:
        """最近看到這家手切的樣子了嗎。"""
        return self._evidence(position, now) >= _NEED[position]

    def tsumogiri(self, position: int, now: float) -> bool | None:
        """剛打的那張是不是摸切。沒量到手牌時回 ``None``,不下判斷。"""
        history = self._seen[position]
        if not history or self._full(position) < _TOO_SMALL[position]:
            return None
        if self.tegiri_seen(position, now):
            return False
        # 開局十四張可能連在一起,尚不知道哪一段才是十三張本體。
        # 從未看過分開的摸牌時,沒看到缺口不能當成摸切的證據。
        if position != 1 and position not in self._drawn_seen:
            return None
        return True

    def _full(self, position: int) -> float:
        """手牌原本有多長:最近看到最完整的那一次。沿線量的只算沒被手切開的。"""
        bodies = [
            shape.body for _, shape in self._seen[position] if position == 1 or shape.clean
        ]
        return max(bodies, default=0.0)

    def _evidence(self, position: int, now: float) -> int:
        full = self._full(position)
        recent = [shape for at, shape in self._seen[position] if at >= now - _RECENT]
        if position == 1:
            return sum(s.pieces >= 2 and s.body >= _WHOLE * full for s in recent)
        shorter = _SHORTER[position]
        return sum(s.drawn and s.clean and s.body <= full - shorter for s in recent)
