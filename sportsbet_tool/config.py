from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sportsbet_tool.risk import RiskConfig


@dataclass(slots=True)
class AppConfig:
    bankroll_amount: float = 1000.0
    currency: str = "USD"
    default_bookmaker: str = "bet365"
    risk: RiskConfig = field(default_factory=RiskConfig)
    jurisdiction_confirmed: bool = False
    allow_final_submit_automation: bool = False
    web_bridge_endpoint: str = "http://127.0.0.1:10086/command"
    web_bridge_session: str = "sportsbet-tool"


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()
    data = _load_mapping(Path(path))
    bankroll = data.get("bankroll", {})
    markets = data.get("markets", {})
    risk = data.get("risk", {})
    compliance = data.get("compliance", {})
    web_bridge = data.get("web_bridge", {})
    return AppConfig(
        bankroll_amount=float(bankroll.get("starting_amount", 1000.0)),
        currency=bankroll.get("currency", "USD"),
        default_bookmaker=markets.get("default_bookmaker", "bet365"),
        risk=RiskConfig(
            kelly_fraction=float(risk.get("kelly_fraction", 0.5)),
            max_single_bet_fraction=float(risk.get("max_single_bet_fraction", 0.02)),
            max_daily_risk_fraction=float(risk.get("max_daily_risk_fraction", 0.05)),
            min_edge=float(risk.get("min_edge", 0.015)),
            pause_after_consecutive_losses=int(risk.get("pause_after_consecutive_losses", 5)),
        ),
        jurisdiction_confirmed=bool(compliance.get("jurisdiction_confirmed", False)),
        allow_final_submit_automation=bool(compliance.get("allow_final_submit_automation", False)),
        web_bridge_endpoint=web_bridge.get("endpoint", "http://127.0.0.1:10086/command"),
        web_bridge_session=web_bridge.get("session", "sportsbet-tool"),
    )


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install PyYAML or provide a JSON config file.") from exc
