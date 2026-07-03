from __future__ import annotations

from dataclasses import asdict
from typing import Any

from sportsbet_tool.cli import _build_portfolio_planner, _build_recommendations, _load_example_bundle
from sportsbet_tool.config import load_config
from sportsbet_tool.intent import parse_intent
from sportsbet_tool.models import clean_dict
from sportsbet_tool.odds import no_vig_probabilities_by_market
from sportsbet_tool.quality import FeatureQualityAnalyzer
from sportsbet_tool.risk import RiskMode, risk_config_for_mode


def predict_sample(
    bankroll: float = 1000.0,
    config_path: str | None = None,
    active_only: bool = False,
    risk_mode: RiskMode | None = None,
) -> dict[str, Any]:
    """Agent-friendly JSON API for running the bundled prediction scenario."""

    config = load_config(config_path)
    if risk_mode:
        config.risk = risk_config_for_mode(risk_mode)
    recommendations = _build_recommendations(_load_example_bundle(), bankroll, config)
    if active_only:
        recommendations = [item for item in recommendations if item.status == "recommended"]
    return {
        "bookmaker": config.default_bookmaker,
        "bankroll": bankroll,
        "risk_mode": config.risk.mode,
        "recommendations": [clean_dict(item) for item in recommendations],
        "manual_confirmation_required": True,
    }


def backtest_sample(
    bankroll: float = 1000.0,
    config_path: str | None = None,
    risk_mode: RiskMode | None = None,
) -> dict[str, Any]:
    """Agent-friendly JSON API for running the bundled backtest scenario."""

    # Keep a stable direct API without shelling out; call the same pipeline as the CLI.
    config = load_config(config_path)
    if risk_mode:
        config.risk = risk_config_for_mode(risk_mode)
    bundle = _load_example_bundle()
    from sportsbet_tool.backtesting import BacktestEngine, BacktestSample
    from sportsbet_tool.ensemble import ModelDebatePredictor
    from sportsbet_tool.features import FeatureBuilder
    from sportsbet_tool.risk import BankrollStrategy

    builder = FeatureBuilder()
    quality = FeatureQualityAnalyzer()
    fair_probabilities = no_vig_probabilities_by_market(bundle["odds"].values())
    samples: list[BacktestSample] = []
    for row in bundle["historical_results"]:
        match = bundle["matches"][row["match_id"]]
        odds = bundle["odds"][row["market_id"]]
        vector = builder.build(
            match=match,
            odds=odds,
            player_ratings=bundle["ratings"],
            esports_styles=bundle["styles"],
            patch_meta=bundle["patch_meta_by_match"].get(match.match_id),
            football_context=bundle["football_context_by_match"].get(match.match_id),
        )
        quality_report = quality.score(vector)
        samples.append(
            BacktestSample(
                start_time=match.start_time,
                vector=vector,
                odds=odds,
                actual_side=row["actual_side"],
                closing_decimal_odds=row.get("closing_decimal_odds"),
                fair_implied_probability=fair_probabilities.get(odds.market_id or ""),
                quality_score=quality_report.score,
                quality_reasons=quality_report.reasons,
            )
        )
    result = BacktestEngine(
        model=ModelDebatePredictor(),
        strategy=BankrollStrategy(config.risk),
        planner=_build_portfolio_planner(config),
    ).run(samples, bankroll)
    payload = asdict(result)
    payload["bookmaker"] = config.default_bookmaker
    payload["risk_mode"] = config.risk.mode
    payload["manual_confirmation_required"] = True
    return payload


def run_intent(
    text: str,
    bankroll: float | None = None,
    config_path: str | None = None,
    risk_mode: RiskMode | None = None,
) -> dict[str, Any]:
    """Infer tool settings from a natural-language request."""

    intent = parse_intent(text)
    config = load_config(config_path)
    selected_bankroll = bankroll if bankroll is not None else intent.bankroll
    selected_bankroll = selected_bankroll if selected_bankroll is not None else config.bankroll_amount
    selected_risk_mode = risk_mode or intent.risk_mode
    if intent.command == "backtest":
        payload = backtest_sample(bankroll=selected_bankroll, config_path=config_path, risk_mode=selected_risk_mode)
    else:
        payload = predict_sample(
            bankroll=selected_bankroll,
            config_path=config_path,
            active_only=intent.active_only,
            risk_mode=selected_risk_mode,
        )
    payload["interpreted_intent"] = {
        "command": intent.command,
        "bankroll": selected_bankroll,
        "risk_mode": selected_risk_mode,
        "active_only": intent.active_only,
        "reasons": intent.reasons,
    }
    return payload
