"""跨次啟動記住的 UI 狀態。

重點只有一個:**這個檔案壞掉不能讓 UI 開不起來**。裡面存的東西弄丟了,
最壞的情況是 Overlay 回到預設位置 —— 為了那個讓程式炸掉是不划算的交易。
"""

from __future__ import annotations

import json

import pytest

from mia.ui.state import MIN_HUD_OPACITY, UiState


class TestDefaults:
    def test_a_fresh_state_hides_the_overlay(self) -> None:
        """預設不顯示。Overlay 蓋在遊戲上,該是使用者明確要求的。"""
        assert UiState().overlay_visible is False

    def test_a_fresh_state_is_not_locked(self) -> None:
        """預設不鎖 —— 鎖住的視窗拖不動,第一次開起來就沒辦法擺位置了。"""
        assert UiState().overlay_locked is False

    def test_a_missing_file_gives_defaults(self, tmp_path) -> None:
        state = UiState.load(tmp_path / "沒有這個檔.json")
        assert state == UiState()

    def test_a_missing_file_still_remembers_where_to_save(self, tmp_path) -> None:
        target = tmp_path / "ui.json"
        state = UiState.load(target)
        state.overlay_visible = True
        state.save()
        assert json.loads(target.read_text(encoding="utf-8"))["overlay_visible"] is True


class TestRoundTrip:
    def test_what_goes_in_comes_back(self, tmp_path) -> None:
        target = tmp_path / "ui.json"
        UiState(
            overlay_visible=True,
            overlay_expanded=True,
            overlay_locked=True,
            overlay_pos=(120, 340),
        ).save(target)
        assert UiState.load(target) == UiState(
            overlay_visible=True,
            overlay_expanded=True,
            overlay_locked=True,
            overlay_pos=(120, 340),
        )

    def test_every_setting_comes_back(self, tmp_path) -> None:
        """存得進去讀不回來的話,症狀是每次啟動版本與框選都不見,而且沒有錯誤訊息。"""
        target = tmp_path / "ui.json"
        saved = UiState(
            platform="steam",
            hand_roi=(0.1, 0.2, 0.3, 0.4),
            table_rect=(0.0, 0.1, 1.0, 0.9),
            canvas="1280x720",
            style="愛鳴牌",
            compare_style="不愛鳴牌",
            log_visible=True,
            log_locked=True,
            log_pos=(10, 20),
            log_size=(480, 360),
        )
        saved.save(target)
        assert UiState.load(target) == saved

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("platform", None),
            ("platform", 3),
            ("hand_roi", [0.1, 0.2, 0.3]),
            ("table_rect", ["a", "b", "c", "d"]),
            ("hand_roi", [True, False, True, False]),
            ("log_pos", [1.5, 2.5]),
            ("log_size", "480x360"),
            ("style", 1),
        ],
    )
    def test_a_wrongly_shaped_value_falls_back(self, tmp_path, name, value) -> None:
        target = tmp_path / "ui.json"
        target.write_text(json.dumps({name: value}), encoding="utf-8")
        assert getattr(UiState.load(target), name) == getattr(UiState(), name)

    def test_the_advice_source_defaults_to_packets(self, tmp_path) -> None:
        assert UiState.load(tmp_path / "none.json").advice_vision_only is False

    def test_the_advice_source_comes_back(self, tmp_path) -> None:
        target = tmp_path / "ui.json"
        UiState(advice_vision_only=True).save(target)
        assert UiState.load(target).advice_vision_only is True

    def test_the_position_comes_back_as_a_tuple(self, tmp_path) -> None:
        """JSON 沒有 tuple,存進去是 list —— 讀回來要轉回去,不然
        ``self.move(*pos)`` 那一類的用法會在型別檢查上炸開。"""
        target = tmp_path / "ui.json"
        UiState(overlay_pos=(1, 2)).save(target)
        assert UiState.load(target).overlay_pos == (1, 2)

    def test_the_path_is_not_written_into_the_file(self, tmp_path) -> None:
        """存放位置不是狀態的一部分。寫進去只會在檔案搬家後變成過期的路徑。"""
        target = tmp_path / "ui.json"
        UiState().save(target)
        assert "path" not in json.loads(target.read_text(encoding="utf-8"))

    def test_the_temporary_file_does_not_survive(self, tmp_path) -> None:
        target = tmp_path / "ui.json"
        UiState().save(target)
        assert list(p.name for p in tmp_path.iterdir()) == ["ui.json"]


class TestBrokenFiles:
    """全部都要回預設值,而不是拋例外。"""

    def test_broken_json_gives_defaults(self, tmp_path) -> None:
        target = tmp_path / "ui.json"
        target.write_text("{ 這不是 JSON", encoding="utf-8")
        assert UiState.load(target) == UiState()

    def test_a_json_list_gives_defaults(self, tmp_path) -> None:
        target = tmp_path / "ui.json"
        target.write_text("[1, 2, 3]", encoding="utf-8")
        assert UiState.load(target) == UiState()

    def test_unknown_keys_are_dropped(self, tmp_path) -> None:
        """舊版本留下來的欄位不該讓建構子炸掉。"""
        target = tmp_path / "ui.json"
        target.write_text(
            json.dumps({"overlay_visible": True, "從前的欄位": 1}), encoding="utf-8"
        )
        assert UiState.load(target).overlay_visible is True

    def test_a_wrongly_typed_flag_falls_back(self, tmp_path) -> None:
        target = tmp_path / "ui.json"
        target.write_text(json.dumps({"overlay_visible": "yes"}), encoding="utf-8")
        assert UiState.load(target).overlay_visible is False

    def test_the_canvas_choice_comes_back(self, tmp_path) -> None:
        state = UiState.load(tmp_path / "ui.json")
        state.canvas = "1920x1080"
        state.save()
        assert UiState.load(tmp_path / "ui.json").canvas == "1920x1080"

    def test_a_wrongly_typed_canvas_falls_back_to_auto(self, tmp_path) -> None:
        """內容看不懂是 Canvas.parse 的事,型別不對則在這裡就該擋掉。"""
        target = tmp_path / "ui.json"
        target.write_text(json.dumps({"canvas": 1080}), encoding="utf-8")
        assert UiState.load(target).canvas is None

    def test_a_wrongly_shaped_position_falls_back(self, tmp_path) -> None:
        """三個數字的座標拿去 ``move(*pos)`` 會直接 TypeError。"""
        target = tmp_path / "ui.json"
        target.write_text(json.dumps({"overlay_pos": [1, 2, 3]}), encoding="utf-8")
        assert UiState.load(target).overlay_pos is None

    def test_a_position_made_of_strings_falls_back(self, tmp_path) -> None:
        target = tmp_path / "ui.json"
        target.write_text(json.dumps({"overlay_pos": ["12", "34"]}), encoding="utf-8")
        assert UiState.load(target).overlay_pos is None

    def test_saving_into_an_unwritable_place_does_not_raise(self, tmp_path) -> None:
        """存不了就算了 —— 使用者正在打牌,不該為了記位置被中斷。"""
        blocker = tmp_path / "檔案"
        blocker.write_text("", encoding="utf-8")
        UiState().save(blocker / "ui.json")


class TestHudOpacity:
    def test_it_round_trips(self, tmp_path) -> None:
        state = UiState.load(tmp_path / "ui.json")
        state.hud_opacity = 0.65
        state.save()
        assert UiState.load(tmp_path / "ui.json").hud_opacity == pytest.approx(0.65)

    @pytest.mark.parametrize(("raw", "expected"), [(0, MIN_HUD_OPACITY), (5, 1.0), (1, 1.0)])
    def test_out_of_range_is_clamped(self, tmp_path, raw, expected) -> None:
        """手改成 0 的人要的是「最透明」,不是預設的不透明。"""
        target = tmp_path / "ui.json"
        target.write_text(json.dumps({"hud_opacity": raw}), encoding="utf-8")
        assert UiState.load(target).hud_opacity == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["半透明", True, None, [0.5]])
    def test_a_wrong_type_gives_the_default(self, tmp_path, raw) -> None:
        target = tmp_path / "ui.json"
        target.write_text(json.dumps({"hud_opacity": raw}), encoding="utf-8")
        assert UiState.load(target).hud_opacity == 1.0


class TestDismissedNotices:
    def test_they_round_trip(self, tmp_path) -> None:
        state = UiState.load(tmp_path / "ui.json")
        state.dismissed_notices = ("yolo_skin",)
        state.save()
        assert UiState.load(tmp_path / "ui.json").dismissed_notices == ("yolo_skin",)

    @pytest.mark.parametrize("raw", ["yolo_skin", [1], None, {"yolo_skin": True}])
    def test_a_wrong_type_shows_every_notice_again(self, tmp_path, raw) -> None:
        """壞掉的值當成沒勾過 —— 多跳一次提醒,比該跳的時候不跳好。"""
        target = tmp_path / "ui.json"
        target.write_text(json.dumps({"dismissed_notices": raw}), encoding="utf-8")
        assert UiState.load(target).dismissed_notices == ()
