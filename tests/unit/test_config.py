"""設定載入與驗證測試。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from majsoul_copilot.config.loader import _deep_merge, load_config, save_config
from majsoul_copilot.config.models import AppConfig, CalibrationConfig, CaptureConfig


class TestDeepMerge:
    def test_nested_dicts_merge_recursively(self) -> None:
        base = {"capture": {"backend": "auto", "window": {"min_width": 640}}}
        override = {"capture": {"window": {"min_width": 800}}}
        assert _deep_merge(base, override) == {
            "capture": {"backend": "auto", "window": {"min_width": 800}}
        }

    def test_lists_are_replaced_not_appended(self) -> None:
        """設定裡的 list 語意是「完整替換」,不是「附加」。"""
        merged = _deep_merge({"patterns": ["a", "b"]}, {"patterns": ["c"]})
        assert merged["patterns"] == ["c"]

    def test_does_not_mutate_inputs(self) -> None:
        base = {"a": {"b": 1}}
        _deep_merge(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}


class TestDefaults:
    def test_loads_project_default_yaml(self) -> None:
        config = load_config()
        assert config.capture.backend == "auto"
        assert config.capture.retina is True
        assert config.calibration.aspect_ratio == pytest.approx(16 / 9, rel=1e-6)

    def test_default_patterns_cover_browser_and_client(self) -> None:
        window = load_config().capture.window
        assert any("雀魂" in p for p in window.title_patterns)
        assert any("majsoul" in p.lower() for p in window.owner_patterns)

    def test_model_defaults_work_without_any_file(self) -> None:
        config = AppConfig()
        assert config.capture.target_fps > 0
        assert config.calibration.manual_table_rect is None


class TestValidation:
    def test_unknown_key_is_rejected(self) -> None:
        """打錯的鍵名必須報錯,不能被默默忽略 —— 否則設定沒生效卻毫無徵兆。"""
        with pytest.raises(ValidationError):
            load_config(overrides={"capture": {"retna": True}})

    def test_unknown_backend_rejected(self) -> None:
        with pytest.raises(ValidationError):
            load_config(overrides={"capture": {"backend": "directx"}})

    def test_target_fps_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            CaptureConfig(target_fps=0)

    def test_aspect_tolerance_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            CalibrationConfig(aspect_tolerance=0)

    def test_config_is_frozen(self) -> None:
        config = AppConfig()
        with pytest.raises(ValidationError):
            config.capture.retina = False  # type: ignore[misc]


class TestOverrides:
    def test_overrides_take_precedence(self) -> None:
        config = load_config(overrides={"capture": {"backend": "mss"}})
        assert config.capture.backend == "mss"

    def test_override_does_not_clobber_siblings(self) -> None:
        config = load_config(overrides={"capture": {"backend": "mss"}})
        assert config.capture.window.min_width == 640  # 仍是 default.yaml 的值


class TestRoundtrip:
    def test_save_then_load(self, tmp_path: Path) -> None:
        original = AppConfig(capture=CaptureConfig(backend="mss", target_fps=30.0))
        target = tmp_path / "out.yaml"
        save_config(original, target)

        reloaded = load_config(target)
        assert reloaded.capture.backend == "mss"
        assert reloaded.capture.target_fps == 30.0

    def test_missing_file_is_not_an_error(self, tmp_path: Path) -> None:
        """local 覆寫檔不存在是正常情況,不該炸掉。"""
        assert load_config(tmp_path / "nope.yaml") is not None

    def test_non_mapping_yaml_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("- 這是一個 list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="必須是 mapping"):
            load_config(bad)
