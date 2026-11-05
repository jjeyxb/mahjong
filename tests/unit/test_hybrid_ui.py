from mia.features import RecognitionChoice
from mia.ui.panel.window import PanelWindow
from mia.ui.state import UiState
from mia.ui.viewmodel import ViewModel


def test_hybrid_picker_uses_claudes_existing_three_page_synchronization(qtbot):
    class Picker:
        def __init__(self):
            self.choice = RecognitionChoice()
            self.changes = 0

        def can_pick_recognition(self):
            return True

        def recognition_backend(self):
            return self.choice.value

        def set_recognition_backend(self, value):
            self.changes += 1
            self.choice.set(value)

    picker = Picker()
    window = PanelWindow(ViewModel(), recognition=picker)
    qtbot.addWidget(window)
    choices = window._recognition_pickers  # noqa: SLF001
    assert len(choices) == 3
    choices[1].setCurrentIndex(choices[1].findData("hybrid"))
    assert picker.changes == 1
    assert all(c.currentData() == "hybrid" for c in choices)


def test_hybrid_backend_setting_survives_restart(tmp_path):
    path = tmp_path / "state.json"
    UiState(recognition_backend="hybrid").save(path)
    assert RecognitionChoice(UiState.load(path).recognition_backend).value == "hybrid"
