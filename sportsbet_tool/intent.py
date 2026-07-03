from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from sportsbet_tool.risk import RiskMode

IntentCommand = Literal["predict", "backtest"]


@dataclass(slots=True)
class IntentSettings:
    command: IntentCommand = "predict"
    bankroll: float | None = None
    risk_mode: RiskMode = "steady"
    active_only: bool = True
    reasons: list[str] = field(default_factory=list)


INSURANCE_HINTS = (
    "\u4fdd\u5b88",
    "\u7a33",
    "\u8c28\u614e",
    "\u5c0f\u6ce8",
    "\u4f4e\u98ce\u9669",
    "\u4fdd\u9669",
    "\u522b\u5192\u9669",
    "\u5c11\u4e0b",
    "conservative",
    "safe",
)
ADVENTUROUS_HINTS = (
    "\u6fc0\u8fdb",
    "\u8fdb\u53d6",
    "\u5927\u80c6",
    "\u9ad8\u98ce\u9669",
    "\u591a\u4e0b",
    "\u5192\u9669",
    "adventurous",
    "aggressive",
)
WILD_HINTS = ("\u68ad\u54c8", "\u75af\u72c2", "\u6700\u5927", "\u62c9\u6ee1", "all in", "all-in", "wild")
BACKTEST_HINTS = ("\u56de\u6d4b", "\u5386\u53f2", "\u8868\u73b0", "\u9a8c\u8bc1", "\u590d\u76d8", "backtest", "bt")
ALL_HINTS = ("\u5168\u90e8", "\u5b8c\u6574", "\u6240\u6709", "all", "show all")
ACTIVE_HINTS = ("\u53ea\u770b\u63a8\u8350", "\u6709\u4ef7\u503c", "\u503c\u5f97", "active")


def parse_intent(text: str | None) -> IntentSettings:
    raw = (text or "").strip()
    lowered = raw.lower()
    settings = IntentSettings()
    if not raw:
        settings.reasons.append("default_active_steady")
        return settings

    if _has_any(lowered, BACKTEST_HINTS):
        settings.command = "backtest"
        settings.reasons.append("intent_backtest")

    if _has_any(lowered, WILD_HINTS):
        settings.risk_mode = "wild"
        settings.reasons.append("risk_wild_hint")
    elif _has_any(lowered, ADVENTUROUS_HINTS):
        settings.risk_mode = "adventurous"
        settings.reasons.append("risk_adventurous_hint")
    elif _has_any(lowered, INSURANCE_HINTS):
        settings.risk_mode = "insurance"
        settings.reasons.append("risk_insurance_hint")
    else:
        settings.risk_mode = "steady"
        settings.reasons.append("risk_default_steady")

    settings.bankroll = _extract_bankroll(lowered)
    if settings.bankroll is not None:
        settings.reasons.append("bankroll_from_text")

    if _has_any(lowered, ALL_HINTS):
        settings.active_only = False
        settings.reasons.append("show_all_hint")
    elif _has_any(lowered, ACTIVE_HINTS):
        settings.active_only = True
        settings.reasons.append("active_only_hint")
    else:
        settings.active_only = True
        settings.reasons.append("active_only_default")

    return settings


def _has_any(text: str, hints: tuple[str, ...]) -> bool:
    return any(hint in text for hint in hints)


def _extract_bankroll(text: str) -> float | None:
    money_words = "|".join(
        [
            "\u672c\u91d1",
            "\u8d44\u91d1",
            "\u9884\u7b97",
            "\u8d26\u6237",
            "\u7528",
            "bankroll",
            "budget",
        ]
    )
    patterns = [
        rf"(?:{money_words})\s*[:\uff1a]?\s*[$\uffe5\u00a5]?\s*(\d+(?:\.\d+)?)\s*(\u4e07|k|\u5343|usdt|usd)?",
        r"[$]\s*(\d+(?:\.\d+)?)\s*(k|usdt|usd)?",
        rf"(\d+(?:\.\d+)?)\s*(\u4e07|k|\u5343|usdt|usd)?\s*(?:{money_words})",
        r"(\d+(?:\.\d+)?)\s*(usdt|usd)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        amount = float(match.group(1))
        unit = match.group(2)
        if unit == "\u4e07":
            amount *= 10000
        elif unit in {"k", "\u5343"}:
            amount *= 1000
        return amount
    return None
