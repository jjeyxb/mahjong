"""設定載入:YAML → :class:`AppConfig`。

載入順序(後者覆蓋前者,採遞迴合併):

1. 內建預設值(定義在 :mod:`mia.config.models`)
2. ``config/default.yaml`` —— 進版控,團隊共用
3. ``config/default.local.yaml`` —— 不進版控,個人機器的覆寫
4. 明確傳入的 ``path``
5. 程式碼傳入的 ``overrides``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mia.config.models import AppConfig
from mia.utils.paths import CONFIG_DIR

__all__ = ["DEFAULT_CONFIG_PATH", "LOCAL_CONFIG_PATH", "load_config", "save_config"]

DEFAULT_CONFIG_PATH: Path = CONFIG_DIR / "default.yaml"
LOCAL_CONFIG_PATH: Path = CONFIG_DIR / "default.local.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """遞迴合併兩層以上的巢狀 dict。

    只有「兩邊都是 dict」時才往下合併;其餘情況(含 list)一律整個取代。
    這是刻意的 —— 設定裡的 list(例如 title_patterns)語意上是「完整替換」,
    而不是「附加」。
    """
    result = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"設定檔最外層必須是 mapping,但 {path} 是 {type(data).__name__}")
    return data


def load_config(
    path: Path | str | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """載入並驗證設定。

    Args:
        path: 額外的設定檔;None 時只使用 default.yaml 與 default.local.yaml。
        overrides: 最高優先權的覆寫,結構須與 YAML 相同。

    Raises:
        pydantic.ValidationError: 設定內容不合法(含未知欄位 —— 模型設了
            ``extra="forbid"``,打錯的鍵名會直接報錯而不是被默默忽略)。
    """
    merged: dict[str, Any] = {}
    for candidate in (DEFAULT_CONFIG_PATH, LOCAL_CONFIG_PATH):
        merged = _deep_merge(merged, _read_yaml(candidate))
    if path is not None:
        merged = _deep_merge(merged, _read_yaml(Path(path)))
    if overrides:
        merged = _deep_merge(merged, overrides)
    return AppConfig.model_validate(merged)


def save_config(config: AppConfig, path: Path | str) -> None:
    """把設定寫回 YAML。ROI 標註工具校正完之後會用到。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            config.model_dump(mode="json"),
            fh,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
