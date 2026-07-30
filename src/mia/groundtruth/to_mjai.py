"""雀魂 liqi 動作 → MJAI 事件流。

這是一台狀態機。兩套協定的模型並不對齊,轉換必須補上雀魂沒有明說的資訊:

* 雀魂發給莊家 14 張牌、其他人 13 張;MJAI 是所有人 13 張,再補一個 ``tsumo``。
* 立直在雀魂只是宣言牌上的一個旗標;MJAI 是 ``reach`` → ``dahai`` →
  (宣言牌安全通過後)``reach_accepted`` 三個事件。
* 榮和的放銃者不在 ``HuleInfo`` 裡,要靠追蹤「最後一個亮出牌的人」才能還原
  —— 而且不能只看切牌,搶槓、國士搶暗槓、三麻搶北都會讓答案不同。
* 槓的寶牌翻牌時機,暗槓與加槓/大明槓**不一樣**。

參考資料
--------
協定語意主要參考 Akagi 專案的 Majsoul bridge 說明文件
(https://github.com/shinkuan/Akagi,Apache-2.0)。該文件把幾個容易做錯的地方
記錄得很清楚,尤其是寶牌翻牌時機與 ``reach_accepted`` 的插入位置。
本模組是依該文件重新以 Python 實作,並未複製其 Rust 程式碼;
授權聲明見專案根目錄的 ``NOTICE``。

.. warning::
   **尚未以真實流量驗證。** 欄位名稱已對照專案內附的 liqi.json 逐一確認,
   單元測試也涵蓋各分支,但實際對局中的邊界情況(多人榮和、三麻、
   斷線重連)仍需錄一場真實對局跑過才算數。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from mia.groundtruth.liqi import LiqiMessage, LiqiParser, MessageKind
from mia.mjai.events import (
    Ankan,
    Chi,
    Dahai,
    Daiminkan,
    Dora,
    EndGame,
    EndKyoku,
    Hora,
    Kakan,
    Kita,
    MjaiEvent,
    Pon,
    Reach,
    ReachAccepted,
    Ryukyoku,
    StartGame,
    StartKyoku,
    Tsumo,
)
from mia.mjai.tiles import UNKNOWN, TileError, is_red_five, ms_to_mjai
from mia.utils.logging import logger

__all__ = ["DoraTiming", "MajsoulToMjai"]

_BAKAZE = ("E", "S", "W", "N")

_AUTH_GAME = ".lq.FastTest.authGame"
_SYNC_GAME = ".lq.FastTest.syncGame"
_ENTER_GAME = ".lq.FastTest.enterGame"
_GAME_END = ".lq.NotifyGameEndResult"

_A_NEW_ROUND = ".lq.ActionNewRound"
_A_DEAL_TILE = ".lq.ActionDealTile"
_A_DISCARD = ".lq.ActionDiscardTile"
_A_CHI_PENG_GANG = ".lq.ActionChiPengGang"
_A_ANGANG_ADDGANG = ".lq.ActionAnGangAddGang"
_A_HULE = ".lq.ActionHule"
_A_NO_TILE = ".lq.ActionNoTile"
_A_LIUJU = ".lq.ActionLiuJu"
_A_BABEI = ".lq.ActionBaBei"


class DoraTiming(Enum):
    """槓之後,新寶牌指示牌該在什麼時候翻開。

    麻將規則上暗槓是「即乗り」(立刻翻),加槓與大明槓是「後乗り」(打出牌後才翻)。
    雀魂只是把新的指示牌塞在嶺上摸牌那則 ``ActionDealTile`` 的 ``doras`` 裡,
    沒有告訴你該什麼時候翻,所以要自己記住剛才槓的種類。

    ==========  ==========================================
    槓的種類     事件順序
    ==========  ==========================================
    暗槓         ``ankan`` → ``dora`` → 嶺上 ``tsumo`` → ``dahai``
    加槓         ``kakan`` → 嶺上 ``tsumo`` → ``dora`` → ``dahai``
    大明槓       ``daiminkan`` → 嶺上 ``tsumo`` → ``dora`` → ``dahai``
    ==========  ==========================================
    """

    NONE = auto()
    BEFORE_RINSHAN = auto()  # 暗槓:嶺上摸牌之前翻
    AFTER_RINSHAN = auto()  # 加槓 / 大明槓:打牌之前才翻


@dataclass
class _GameState:
    """一場對局的轉換狀態。每次 ``start_game`` 重建。"""

    seat: int = 0
    num_players: int = 4
    names: list[str] = field(default_factory=list)

    # --- 每局重置 ---
    doras: list[str] = field(default_factory=list)
    dora_timing: DoraTiming = DoraTiming.NONE
    deferred_dora: str | None = None
    pending_reach: int | None = None
    last_revealed_by: int | None = None

    def reset_kyoku(self) -> None:
        self.doras.clear()
        self.dora_timing = DoraTiming.NONE
        self.deferred_dora = None
        self.pending_reach = None
        self.last_revealed_by = None


class MajsoulToMjai:
    """把 liqi 訊息流轉成 MJAI 事件流。

    一場對局用一個實例。用法::

        converter = MajsoulToMjai(parser)
        for frame, message in parse_dump(path, schema):
            for event in converter.handle(message):
                print(json.dumps(event.to_dict()))
    """

    def __init__(self, parser: LiqiParser) -> None:
        # 需要 parser 是為了解 GameRestore 裡夾帶的動作(它們沒有經過混淆)
        self._parser = parser
        self._state = _GameState()
        self._account_id: int | None = None
        self._started = False

    @property
    def state(self) -> _GameState:
        return self._state

    @property
    def seat(self) -> int:
        """自己的座位。``authGame`` 之前為 0。"""
        return self._state.seat

    # ------------------------------------------------------------------ 入口

    def handle(self, message: LiqiMessage) -> list[MjaiEvent]:
        """處理一則 liqi 訊息,回傳它產生的 MJAI 事件(可能是空的)。"""
        if message.method == _AUTH_GAME:
            if message.kind is MessageKind.REQUEST:
                self._account_id = getattr(message.payload, "account_id", None)
                return []
            return self._on_auth_game(message.payload)

        if message.method in (_SYNC_GAME, _ENTER_GAME) and message.kind is MessageKind.RESPONSE:
            return self._on_game_restore(message.payload)

        if message.method == _GAME_END:
            return [EndGame()]

        if message.is_action:
            return self._on_action(message.action_name, message.action)

        return []

    # ------------------------------------------------------------------ 對局開始

    def _on_auth_game(self, payload: Any) -> list[MjaiEvent]:
        """從 authGame 的回應解析座位、人數與玩家名稱。"""
        seat_list = list(getattr(payload, "seat_list", []))
        if not seat_list:
            logger.warning("authGame 回應沒有 seat_list,無法確定座位")
            return []

        num_players = len(seat_list)
        if num_players not in (3, 4):
            logger.warning("seat_list 長度為 {},非 3 或 4,退回以 4 人處理", num_players)
            num_players = 4

        seat = seat_list.index(self._account_id) if self._account_id in seat_list else 0
        if self._account_id not in seat_list:
            logger.warning("account_id {} 不在 seat_list 中,座位退回 0", self._account_id)

        # 機器人在 robots[] 底下且沒有暱稱,補空字串以維持長度與座位一致
        nicknames = {
            p.account_id: p.nickname
            for p in list(getattr(payload, "players", []))
            if getattr(p, "nickname", "")
        }
        names = [nicknames.get(account_id, "") for account_id in seat_list]

        self._state = _GameState(seat=seat, num_players=num_players, names=names)
        self._started = True
        return [StartGame(id=seat, names=names)]

    def _on_game_restore(self, payload: Any) -> list[MjaiEvent]:
        """斷線重連:重播 ``GameRestore.actions[]`` 還原目前這一局。

        重連時雀魂會先重跑一次 ``authGame``(已經發過 ``start_game``),再送一份
        本局至今的完整動作紀錄。這裡把每個動作餵回一般的處理流程,
        所有轉換邏輯與立直記帳都原樣重用,不另寫一套。

        **這些動作沒有經過 XOR 混淆** —— 與即時的 ``ActionPrototype`` 不同,
        對它們套反混淆反而會把位元組弄壞。
        """
        restore = getattr(payload, "game_restore", None)
        actions = list(getattr(restore, "actions", [])) if restore is not None else []
        if not actions:
            return []

        logger.info("斷線重連:重播 {} 個動作以還原本局", len(actions))
        events: list[MjaiEvent] = []
        for prototype in actions:
            name, action, _ = self._parser.parse_action_prototype(prototype, obfuscated=False)
            if action is not None:
                events.extend(self._on_action(name, action))
        return events

    # ------------------------------------------------------------------ 動作分派

    def _on_action(self, name: str, action: Any) -> list[MjaiEvent]:
        """處理一個牌局動作,並在此統一處理立直成立的插入。"""
        events: list[MjaiEvent] = []
        state = self._state

        # 立直宣言牌通過之後才記點。被榮和則立直作廢,不發 reach_accepted。
        # 宣言後緊接著又是切牌在實務上不可能發生(同一人不會連打兩張),
        # 保險起見保留佇列讓下一個正當動作來處理。
        if state.pending_reach is not None and name not in (_A_HULE, _A_DISCARD):
            events.append(ReachAccepted(actor=state.pending_reach))
            state.pending_reach = None
        elif name == _A_HULE:
            state.pending_reach = None

        handler = {
            _A_NEW_ROUND: self._build_new_round,
            _A_DEAL_TILE: self._build_deal_tile,
            _A_DISCARD: self._build_discard,
            _A_CHI_PENG_GANG: self._build_chi_peng_gang,
            _A_ANGANG_ADDGANG: self._build_angang_addgang,
            _A_HULE: self._build_hule,
            _A_NO_TILE: self._build_no_tile,
            _A_LIUJU: self._build_liuju,
            _A_BABEI: self._build_babei,
        }.get(name)

        if handler is None:
            return events  # ActionMJStart 等沒有 MJAI 對應的動作

        try:
            events.extend(handler(action))
        except TileError as exc:
            logger.warning("{} 含無法辨識的牌,略過此事件: {}", name, exc)
        return events

    # ------------------------------------------------------------------ 各動作

    def _build_new_round(self, action: Any) -> list[MjaiEvent]:
        """配牌。雀魂給莊家 14 張,MJAI 要拆成 13 張 + 一個 tsumo。"""
        state = self._state
        state.reset_kyoku()

        oya = int(getattr(action, "ju", 0))
        tiles = [ms_to_mjai(t) for t in getattr(action, "tiles", [])]

        doras = list(getattr(action, "doras", []))
        marker_source = doras[0] if doras else getattr(action, "dora", "")
        if not marker_source:
            logger.warning("ActionNewRound 沒有寶牌指示牌")
            dora_marker = UNKNOWN
        else:
            dora_marker = ms_to_mjai(marker_source)
        state.doras = [dora_marker]

        # 只看得到自己的手牌;其他人一律以 ? 佔位
        hands = [[UNKNOWN] * 13 for _ in range(state.num_players)]
        drawn: str | None = None
        if len(tiles) >= 14:  # 自己是莊家
            hands[state.seat] = sorted(tiles[:13])
            drawn = tiles[13]
        elif tiles:
            hands[state.seat] = sorted(tiles[:13])

        scores = [int(s) for s in getattr(action, "scores", [])][: state.num_players]
        if len(scores) < state.num_players:
            scores += [0] * (state.num_players - len(scores))

        events: list[MjaiEvent] = [
            StartKyoku(
                bakaze=_BAKAZE[int(getattr(action, "chang", 0)) % 4],
                kyoku=oya + 1,
                honba=int(getattr(action, "ben", 0)),
                kyotaku=int(getattr(action, "liqibang", 0)),
                oya=oya,
                dora_marker=dora_marker,
                tehais=hands,
                scores=scores,
            )
        ]
        # 莊家的第一次摸牌:自己是莊家就知道摸到什麼,否則只知道有這件事
        events.append(Tsumo(actor=oya, pai=drawn if drawn is not None else UNKNOWN))
        return events

    def _build_deal_tile(self, action: Any) -> list[MjaiEvent]:
        """摸牌。暗槓後的新寶牌要在這裡、摸牌之前翻。"""
        state = self._state
        events: list[MjaiEvent] = []

        new_doras = self._consume_new_doras(action)
        if new_doras:
            if state.dora_timing is DoraTiming.BEFORE_RINSHAN:
                events.extend(Dora(dora_marker=d) for d in new_doras)
            elif state.dora_timing is DoraTiming.AFTER_RINSHAN:
                state.deferred_dora = new_doras[-1]
                if len(new_doras) > 1:  # 理論上一次只會翻一張
                    events.extend(Dora(dora_marker=d) for d in new_doras[:-1])
            else:
                events.extend(Dora(dora_marker=d) for d in new_doras)
        state.dora_timing = DoraTiming.NONE

        actor = int(getattr(action, "seat", 0))
        tile = getattr(action, "tile", "")
        pai = ms_to_mjai(tile) if (actor == state.seat and tile) else UNKNOWN
        events.append(Tsumo(actor=actor, pai=pai))
        return events

    def _build_discard(self, action: Any) -> list[MjaiEvent]:
        """打牌。立直宣言與延後的寶牌都在這裡處理。"""
        state = self._state
        events: list[MjaiEvent] = []

        # 加槓 / 大明槓的寶牌:在打牌之前才翻
        if state.deferred_dora is not None:
            events.append(Dora(dora_marker=state.deferred_dora))
            state.deferred_dora = None

        actor = int(getattr(action, "seat", 0))
        tile = getattr(action, "tile", "")
        if not tile:
            raise TileError(f"ActionDiscardTile(seat={actor}) 沒有牌")

        declaring_reach = bool(getattr(action, "is_liqi", False)) or bool(
            getattr(action, "is_wliqi", False)
        )
        if declaring_reach:
            events.append(Reach(actor=actor))

        events.append(
            Dahai(
                actor=actor,
                pai=ms_to_mjai(tile),
                tsumogiri=bool(getattr(action, "moqie", False)),
            )
        )

        if declaring_reach:
            state.pending_reach = actor
        state.last_revealed_by = actor
        return events

    def _build_chi_peng_gang(self, action: Any) -> list[MjaiEvent]:
        """吃 / 碰 / 大明槓。type = 0 / 1 / 2。"""
        state = self._state
        meld_type = int(getattr(action, "type", 0))
        actor = int(getattr(action, "seat", 0))
        tiles = [ms_to_mjai(t) for t in getattr(action, "tiles", [])]
        froms = [int(f) for f in getattr(action, "froms", [])]

        if meld_type == 0 and state.num_players == 3:
            logger.warning("三人麻將不應出現吃牌,已忽略")
            return []
        if len(tiles) != len(froms):
            logger.warning("tiles 與 froms 長度不符({} vs {}),略過", len(tiles), len(froms))
            return []

        # 被鳴的那張來自唯一一個非鳴牌者;其餘是自己手上打出去組合的牌
        taken = [(t, f) for t, f in zip(tiles, froms, strict=True) if f != actor]
        consumed = [t for t, f in zip(tiles, froms, strict=True) if f == actor]
        if not taken:
            logger.warning("鳴牌的 froms 全部等於 actor,找不到來源,略過")
            return []
        pai, target = taken[0]

        if meld_type == 0:
            return [Chi(actor=actor, target=target, pai=pai, consumed=consumed)]
        if meld_type == 1:
            return [Pon(actor=actor, target=target, pai=pai, consumed=consumed)]
        if meld_type == 2:
            state.dora_timing = DoraTiming.AFTER_RINSHAN
            return [Daiminkan(actor=actor, target=target, pai=pai, consumed=consumed)]

        logger.warning("未知的鳴牌型別 {},略過", meld_type)
        return []

    def _build_angang_addgang(self, action: Any) -> list[MjaiEvent]:
        """暗槓(type=3)或加槓(type=2)。"""
        state = self._state
        kan_type = int(getattr(action, "type", 0))
        actor = int(getattr(action, "seat", 0))
        raw_tile = getattr(action, "tiles", "")
        if not raw_tile:
            raise TileError(f"ActionAnGangAddGang(seat={actor}) 沒有牌")
        tile = ms_to_mjai(raw_tile)

        if kan_type == 3:  # 暗槓
            state.dora_timing = DoraTiming.BEFORE_RINSHAN
            state.last_revealed_by = actor  # 國士無雙可搶暗槓
            return [Ankan(actor=actor, consumed=_ankan_consumed(tile))]

        if kan_type == 2:  # 加槓
            state.dora_timing = DoraTiming.AFTER_RINSHAN
            state.last_revealed_by = actor  # 搶槓
            return [Kakan(actor=actor, pai=tile, consumed=_kakan_consumed(tile))]

        logger.warning("未知的槓型別 {},略過", kan_type)
        return []

    def _build_babei(self, action: Any) -> list[MjaiEvent]:
        """三麻的拔北。拔北**不會**翻新寶牌。"""
        state = self._state
        actor = int(getattr(action, "seat", 0))
        if state.num_players != 3:
            logger.warning("四人麻將出現拔北(seat={}),仍照常發出", actor)
        state.last_revealed_by = actor  # 搶北
        return [Kita(actor=actor)]

    def _build_hule(self, action: Any) -> list[MjaiEvent]:
        """和了。每個 ``hules`` 條目一個 hora,最後補一個 end_kyoku。"""
        state = self._state
        deltas = [int(d) for d in getattr(action, "delta_scores", [])] or None

        events: list[MjaiEvent] = []
        for hule in getattr(action, "hules", []):
            actor = int(getattr(hule, "seat", 0))
            if getattr(hule, "zimo", False):
                target = actor
            elif state.last_revealed_by is not None:
                target = state.last_revealed_by
            else:
                # 沒有任何亮牌動作卻出現榮和 —— 資料不完整,寧可略過也不要記錯放銃者
                logger.warning("榮和(seat={})之前沒有亮牌動作,無法判定放銃者,略過", actor)
                continue

            ura = (
                [ms_to_mjai(t) for t in getattr(hule, "li_doras", [])]
                if getattr(hule, "liqi", False)
                else None
            )
            events.append(Hora(actor=actor, target=target, deltas=deltas, ura_markers=ura))

        events.append(EndKyoku())
        return events

    def _build_no_tile(self, action: Any) -> list[MjaiEvent]:
        """荒牌流局。把每筆點數變動記錄依座位加總。"""
        num = self._state.num_players
        totals = [0] * num
        found = False
        for record in getattr(action, "scores", []):
            for i, delta in enumerate(getattr(record, "delta_scores", [])):
                if i < num:
                    totals[i] += int(delta)
                    found = True
        return [Ryukyoku(deltas=totals if found else None), EndKyoku()]

    def _build_liuju(self, action: Any) -> list[MjaiEvent]:  # noqa: ARG002
        """途中流局(九種九牌 / 四風連打 / 四家立直 / 四開槓 / 三家和了)。

        ``ActionLiuJu.type`` 在 liqi.proto 裡是沒有 enum 的 uint32,無從得知
        各數值的意義,而 MJAI 的 ``ryukyoku`` 也不區分原因,因此忽略該欄位。
        途中流局沒有點數變動。
        """
        return [Ryukyoku(deltas=None), EndKyoku()]

    # ------------------------------------------------------------------ 工具

    def _consume_new_doras(self, action: Any) -> list[str]:
        """比對動作裡的 ``doras`` 與已知的,回傳這次新增的指示牌。

        雀魂在多個動作上都會重複帶完整的 doras 清單,不是只有新增的那張,
        所以要自己記住看過哪些。
        """
        doras = [ms_to_mjai(t) for t in getattr(action, "doras", [])]
        known = self._state.doras
        if len(doras) <= len(known):
            return []
        new = doras[len(known) :]
        known.extend(new)
        return new


def _ankan_consumed(tile: str) -> list[str]:
    """暗槓的四張牌。

    五萬/五筒/五索的四枚一定包含唯一那張赤五,MJAI 慣例把赤五放在索引 0。
    """
    base = tile[:2] if is_red_five(tile) else tile
    if base in ("5m", "5p", "5s"):
        return [f"{base}r", base, base, base]
    return [tile] * 4


def _kakan_consumed(tile: str) -> list[str]:
    """加槓時原本那副碰的三張牌。

    加上去的若是赤五,原本碰的必然是三張普通五;反之原本那副就含赤五。
    """
    if is_red_five(tile):
        base = tile[:2]
        return [base, base, base]
    if tile in ("5m", "5p", "5s"):
        return [f"{tile}r", tile, tile]
    return [tile] * 3


def events_to_jsonl(events: Iterable[MjaiEvent]) -> str:
    """把事件序列轉成 MJAI 的 JSON Lines 文字。"""
    import json

    return "\n".join(json.dumps(e.to_dict(), ensure_ascii=False) for e in events)
