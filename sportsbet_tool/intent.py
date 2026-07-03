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
    "保守",
    "稳",
    "谨慎",
    "小注",
    "低风险",
    "保险",
    "别冒险",
    "少下",
    "conservative",
    "safe",
)
ADVENTUROUS_HINTS = (
    "激进",
    "进取",
    "大胆",
    "高风险",
    "多下",
    "冒险",
    "adventurous",
    "aggressive",
)
WILD_HINTS = ("梭哈", "疯狂", "最大", "拉满", "all in", "all-in", "wild")
BACKTEST_HINTS = ("回测", "历史", "表现", "验证", "复盘", "backtest", "bt")
ALL_HINTS = ("全部", "完整", "所有", "all", "show all")


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
    elif "只看推荐" in lowered or "有价值" in lowered or "值得" in lowered or "active" in lowered:
        settings.active_only = True
        settings.reasons.append("active_only_hint")
    else:
        settings.active_only = True
        settings.reasons.append("active_only_default")

    return settings


def _has_any(text: str, hints: tuple[str, ...]) -> bool:
    return any(hint in text for hint in hints)


def _extract_bankroll(text: str) -> float | None:
    patterns = [
        r"(?:本金|资金|bankroll|预算|账户|用)\s*[:：]?\s*[$￥¥]?\s*(\d+(?:\.\d+)?)\s*(万|k|千)?",
        r"[$￥¥]\s*(\d+(?:\.\d+)?)\s*(万|k|千)?",
        r"(\d+(?:\.\d+)?)\s*(万|k|千)?\s*(?:本金|资金|预算|bankroll)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        amount = float(match.group(1))
        unit = match.group(2)
        if unit == "万":
            amount *= 10000
        elif unit in {"k", "千"}:
            amount *= 1000
        return amount
    return None
