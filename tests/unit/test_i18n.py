"""介面語言:翻譯檔、模板、設定頁的語言與更新。"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from mia import i18n

ROOT = Path(__file__).resolve().parents[2]
FIELD = re.compile(r"\{(\w+)(?:![rsa])?(?::[^{}]*)?\}")


@pytest.fixture
def english() -> Iterator[None]:
    i18n.set_language("en")
    try:
        yield
    finally:
        i18n.set_language("zh-TW")


def _catalog() -> dict[str, str]:
    return json.loads((i18n.CATALOG_DIR / "en.json").read_text(encoding="utf-8"))


def _code_keys() -> set[str]:
    keys = set()
    for path in [*(ROOT / "src/mia").rglob("*.py"), ROOT / "tools/ui.py"]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "tr"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                keys.add(node.args[0].value)
    return keys


def test_the_default_language_shows_the_source_text() -> None:
    assert i18n.language() == "zh-TW"
    assert i18n.tr("視窗置頂") == "視窗置頂"
    assert i18n.tr("目前版本 {value}", value="0.1.0") == "目前版本 0.1.0"
    assert i18n.translate("擷取失敗 3 次(逾時)") == "擷取失敗 3 次(逾時)"


def test_english_translates_templates_and_composed_messages(english: None) -> None:
    del english
    assert i18n.tr("視窗置頂") == "Keep window on top"
    assert i18n.tr("新版 {version}，下載 {value:.1f} MB", version="0.2", value=12.345) == (
        "New version 0.2, download 12.3 MB"
    )
    # 執行緒組好的句子:比對模板,值也翻(「網頁版」)。
    assert i18n.translate("找不到網頁版的遊戲視窗 —— 雀魂開著嗎?") == (
        "Cannot find the Web game window — is Mahjong Soul open?"
    )
    assert i18n.translate("擷取失敗 3 次(逾時)") == "Capture failed 3 times (逾時)"
    # 沒有譯文的照原樣。
    assert i18n.translate("沒有這一句") == "沒有這一句"
    assert i18n.tr("沒有這一句") == "沒有這一句"


def test_an_unknown_language_falls_back_to_the_source() -> None:
    i18n.set_language("xx")
    try:
        assert i18n.language() == "zh-TW"
    finally:
        i18n.set_language("zh-TW")


def test_every_translation_keeps_the_same_fields() -> None:
    for source, target in _catalog().items():
        assert target, source
        assert sorted(m.group(0) for m in FIELD.finditer(source)) == sorted(
            m.group(0) for m in FIELD.finditer(target)
        ), source


def test_every_interface_text_has_an_english_translation() -> None:
    """加了介面上的字就要補 assets/i18n/en.json,不然英文介面會冒出中文。"""
    missing = sorted(_code_keys() - _catalog().keys())
    assert not missing, missing


def test_event_descriptions_use_english_tile_names(english: None) -> None:
    del english
    from mia.mjai.describe import describe, tile_text
    from mia.mjai.events import Dahai, StartKyoku

    assert tile_text("5pr") == "red 5p"
    assert tile_text("E") == "East"
    kyoku = StartKyoku("S", 2, 1, 0, 0, "3m", [["1m"] * 13] + [["?"] * 13] * 3, [25000] * 4)
    assert describe(kyoku, seat=0).startswith("South 2, 1 honba")
    assert describe(Dahai(actor=1, pai="5s", tsumogiri=True), seat=0) == "Shimocha tsumogiri 5s"


def test_the_language_choice_is_remembered(tmp_path: Path) -> None:
    from mia.ui.state import UiState

    path = tmp_path / "ui.json"
    UiState(language="en").save(path)
    assert UiState.load(path).language == "en"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["language"] = 3
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert UiState.load(path).language == "zh-TW"


class _Preferences:
    def __init__(self, *, busy: bool = False, folders: tuple[str, ...] = ()) -> None:
        self.chosen = "zh-TW"
        self.checked = 0
        self.restarted = 0
        self.cleared = 0
        self._busy = busy
        self._folders = folders

    def language(self) -> str:
        return "zh-TW"

    def set_language(self, code: str) -> None:
        self.chosen = code

    def version(self) -> str:
        return "0.1.0"

    def check_updates(self) -> None:
        self.checked += 1

    def busy(self) -> bool:
        return self._busy

    def restart(self) -> None:
        self.restarted += 1

    def data_folders(self) -> tuple[str, ...]:
        return self._folders

    def clear_data(self) -> None:
        self.cleared += 1


def test_the_settings_page_switches_language_and_offers_a_restart(qtbot) -> None:
    from mia.ui.panel.window import PanelWindow
    from mia.ui.viewmodel import ViewModel

    prefs = _Preferences()
    panel = PanelWindow(ViewModel(), preferences=prefs)
    qtbot.addWidget(panel)
    picker = panel._language_picker  # noqa: SLF001
    restart = panel._restart_button  # noqa: SLF001
    row = panel._restart_row  # noqa: SLF001
    assert "0.1.0" in panel._version_label.text()  # noqa: SLF001
    assert row.isHidden()
    picker.setCurrentIndex(picker.findData("en"))
    assert prefs.chosen == "en"
    assert not row.isHidden()
    restart.click()
    assert prefs.restarted == 1
    picker.setCurrentIndex(picker.findData("zh-TW"))
    assert row.isHidden(), "選回原本的語言就不必重新啟動"
    panel._update_button.click()  # noqa: SLF001
    assert prefs.checked == 1


def test_restarting_while_busy_asks_first(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtWidgets import QMessageBox

    from mia.ui.panel.window import PanelWindow
    from mia.ui.viewmodel import ViewModel

    prefs = _Preferences(busy=True)
    panel = PanelWindow(ViewModel(), preferences=prefs)
    qtbot.addWidget(panel)
    answers = [QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes]
    monkeypatch.setattr(QMessageBox, "question", lambda *_: answers.pop(0))
    panel._on_restart()  # noqa: SLF001
    assert prefs.restarted == 0
    panel._on_restart()  # noqa: SLF001
    assert prefs.restarted == 1


def test_clearing_all_data_asks_first_and_lists_the_folders(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from mia.ui.panel.window import PanelWindow
    from mia.ui.viewmodel import ViewModel

    asked: list[str] = []
    answers = [QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes]

    def question(_parent, _title, text):
        asked.append(text)
        return answers.pop(0)

    monkeypatch.setattr(QMessageBox, "question", question)
    prefs = _Preferences(folders=("/x/Application Support/MIA", "/x/Caches/MIA"))
    panel = PanelWindow(ViewModel(), preferences=prefs)
    qtbot.addWidget(panel)
    button = panel._clear_button  # noqa: SLF001
    assert button.isEnabled()
    button.click()
    assert prefs.cleared == 0
    assert "/x/Caches/MIA" in asked[0]
    button.click()
    assert prefs.cleared == 1


def test_the_source_version_has_nothing_to_clear(qtbot) -> None:
    from mia.ui.panel.window import PanelWindow
    from mia.ui.viewmodel import ViewModel

    panel = PanelWindow(ViewModel(), preferences=_Preferences())
    qtbot.addWidget(panel)
    assert not panel._clear_button.isEnabled()  # noqa: SLF001


def test_clear_data_only_removes_owned_folders(tmp_path: Path) -> None:
    module = _tools_ui()
    owned = tmp_path / "Application Support/MIA"
    (owned / "live").mkdir(parents=True)
    (owned / "live/ws.jsonl").write_text("x")
    stranger = tmp_path / "Documents"
    stranger.mkdir()
    module._clear_data([owned, stranger])  # noqa: SLF001
    assert not owned.exists()
    assert stranger.exists(), "名字對不上的資料夾不刪"


def test_without_preferences_the_section_is_disabled(qtbot) -> None:
    from mia.ui.panel.window import PanelWindow
    from mia.ui.viewmodel import ViewModel

    panel = PanelWindow(ViewModel())
    qtbot.addWidget(panel)
    assert not panel._language_picker.isEnabled()  # noqa: SLF001
    assert not panel._update_button.isEnabled()  # noqa: SLF001


def _tools_ui():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_tools_ui", ROOT / "tools" / "ui.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_tools_ui"] = module
    spec.loader.exec_module(module)
    return module


def test_preferences_save_the_language_and_restart_after_cleanup(tmp_path: Path) -> None:
    """設定頁的語言寫進介面設定;重新啟動只是要求結束,新的由 main 的收尾開。"""
    from mia.ui.state import UiState

    module = _tools_ui()

    class App:
        quits = 0

        def quit(self) -> None:
            self.quits += 1

    class Feature:
        enabled = False

    class Runtime:
        running = False
        feature_list = (Feature(),)

        def game_running(self) -> bool:
            return self.running

    state = UiState.load(tmp_path / "ui.json")
    runtime, app = Runtime(), App()
    prefs = module._Preferences(state, runtime, app)  # noqa: SLF001
    prefs.set_language("en")
    assert UiState.load(tmp_path / "ui.json").language == "en"
    assert not prefs.busy()
    runtime.running = True
    assert prefs.busy()
    opened = []
    prefs.updates = lambda: opened.append(1)
    prefs.check_updates()
    assert opened == [1]
    prefs.restart()
    assert prefs.restart_requested and app.quits == 1
    assert module._Preferences(state, None, app).busy() is False  # noqa: SLF001


def test_relaunch_drops_the_update_health_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from PySide6.QtCore import QProcess

    module = _tools_ui()
    started = []
    monkeypatch.setattr(QProcess, "startDetached", lambda *args: started.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        ["tools/ui.py", "--live", "--update-health", "/x/h.json", "n0nce", "--mortal", "w"],
    )
    module._relaunch()  # noqa: SLF001
    assert started == [(sys.executable, ["tools/ui.py", "--live", "--mortal", "w"])]
