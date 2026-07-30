"""MIA — Mahjong Intelligence Assistant。

名稱與縮寫集中在這裡,而不是散在各個要顯示它的地方 —— 視窗標題、CLI 說明、
日誌前綴都會用到,寫死在每一處的話改一次名就得全域搜尋。
"""

from __future__ import annotations

__all__ = ["APP_ABBR", "APP_NAME", "APP_TITLE"]

#: 完整名稱。用在說明文字、關於畫面。
APP_NAME = "Mahjong Intelligence Assistant"

#: 縮寫。用在需要短的地方(視窗標題、日誌)。
APP_ABBR = "MIA"

#: 視窗標題。短名在前 —— macOS 的 Dock 與工作切換器會截斷長標題。
APP_TITLE = f"{APP_ABBR} — {APP_NAME}"
