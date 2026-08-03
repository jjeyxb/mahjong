"""跨次啟動記住的 UI 狀態。

重點只有一個:**這個檔案壞掉不能讓 UI 開不起來**。裡面存的東西弄丟了,
最壞的情況是 Overlay 回到預設位置 —— 為了那個讓程式炸掉是不划算的交易。
"""

from __future__ import annotations

import json

from mia.ui.state import UiState


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
