from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sportsbet_tool.backtesting import BacktestEngine, BacktestSample
from sportsbet_tool.config import load_config
from sportsbet_tool.data_ingestion import (
    football_context_from_dict,
    load_json,
    match_from_dict,
    odds_from_dict,
    patch_meta_from_dict,
    rating_from_dict,
    style_from_dict,
)
from sportsbet_tool.execution_bridge import Bet365SportsbookAdapter, WebBridgeClient
from sportsbet_tool.features import FeatureBuilder
from sportsbet_tool.modeling import HeuristicProbabilityModel
from sportsbet_tool.models import clean_dict
from sportsbet_tool.odds import no_vig_probabilities_by_market
from sportsbet_tool.quality import FeatureQualityAnalyzer
from sportsbet_tool.risk import BankrollStrategy, RiskMode, risk_config_for_mode

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sports and esports prediction assistant.")
    parser.add_argument("--config", help="Optional JSON/YAML config path.")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo-predict", help="Run predictions on bundled sample data.")
    demo.add_argument("--bankroll", type=float, default=None)
    demo.add_argument("--active-only", action="store_true", help="Only print recommended bets.")
    demo.add_argument("--risk-mode", choices=["insurance", "steady", "adventurous", "wild"], default=None)

    backtest = sub.add_parser("backtest", help="Run bundled historical sample backtest.")
    backtest.add_argument("--bankroll", type=float, default=None)
    backtest.add_argument("--risk-mode", choices=["insurance", "steady", "adventurous", "wild"], default=None)

    slip = sub.add_parser("prepare-slip", help="Open sportsbook page and prepare manual bet-slip instructions.")
    slip.add_argument("--url", required=True)
    slip.add_argument("--stake-selector", default=None)
    slip.add_argument("--bankroll", type=float, default=None)
    slip.add_argument("--risk-mode", choices=["insurance", "steady", "adventurous", "wild"], default=None)

    args = parser.parse_args(argv)
    config = load_config(args.config)
    if getattr(args, "risk_mode", None):
        config.risk = risk_config_for_mode(args.risk_mode)
    bankroll = args.bankroll if getattr(args, "bankroll", None) is not None else config.bankroll_amount

    if args.command == "demo-predict":
        return _demo_predict(bankroll, config, active_only=args.active_only)
    if args.command == "backtest":
        return _backtest(bankroll, config)
    if args.command == "prepare-slip":
        return _prepare_slip(args.url, args.stake_selector, bankroll, config)
    return 2


def _demo_predict(bankroll: float, config: Any, active_only: bool = False) -> int:
    bundle = _load_example_bundle()
    recommendations = _build_recommendations(bundle, bankroll, config)
    if active_only:
        recommendations = [item for item in recommendations if item.status == "recommended"]
    print(json.dumps([clean_dict(item) for item in recommendations], indent=2, ensure_ascii=False))
    return 0


def _backtest(bankroll: float, config: Any) -> int:
    bundle = _load_example_bundle()
    fair_probabilities = no_vig_probabilities_by_market(bundle["odds"].values())
    samples: list[BacktestSample] = []
    builder = FeatureBuilder()
    quality = FeatureQualityAnalyzer()
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
    result = BacktestEngine(strategy=BankrollStrategy(config.risk)).run(samples, bankroll)
    payload = asdict(result)
    print(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
    return 0


def _prepare_slip(url: str, stake_selector: str | None, bankroll: float, config: Any) -> int:
    if config.allow_final_submit_automation:
        raise RuntimeError("Final submit automation is intentionally unsupported.")
    bundle = _load_example_bundle()
    recommendations = _build_recommendations(bundle, bankroll, config)
    client = WebBridgeClient(endpoint=config.web_bridge_endpoint, session=config.web_bridge_session)
    slip = Bet365SportsbookAdapter(client).prepare_bet_slip(url, recommendations, stake_selector)
    print(json.dumps(asdict(slip), indent=2, default=str, ensure_ascii=False))
    return 0


def _build_recommendations(bundle: dict[str, Any], bankroll: float, config: Any) -> list[Any]:
    builder = FeatureBuilder()
    model = HeuristicProbabilityModel()
    strategy = BankrollStrategy(config.risk)
    quality = FeatureQualityAnalyzer()
    recommendations = []
    daily_risk = 0.0
    current_day: str | None = None
    default_odds = [odds for odds in bundle["odds"].values() if odds.bookmaker == config.default_bookmaker]
    fair_probabilities = no_vig_probabilities_by_market(default_odds)
    for odds in sorted(default_odds, key=lambda item: bundle["matches"][item.match_id].start_time):
        if odds.bookmaker != config.default_bookmaker:
            continue
        match = bundle["matches"][odds.match_id]
        match_day = match.start_time.date().isoformat()
        if match_day != current_day:
            current_day = match_day
            daily_risk = 0.0
        vector = builder.build(
            match=match,
            odds=odds,
            player_ratings=bundle["ratings"],
            esports_styles=bundle["styles"],
            patch_meta=bundle["patch_meta_by_match"].get(match.match_id),
            football_context=bundle["football_context_by_match"].get(match.match_id),
        )
        quality_report = quality.score(vector)
        prediction = model.predict_vector(vector)
        recommendation = strategy.recommend(
            prediction,
            odds,
            bankroll,
            daily_risk_used=daily_risk,
            fair_implied_probability=fair_probabilities.get(odds.market_id or ""),
            quality_score=quality_report.score,
            quality_reasons=quality_report.reasons,
        )
        if recommendation.status == "recommended":
            daily_risk += recommendation.stake
        recommendations.append(recommendation)
    return recommendations


def _load_example_bundle() -> dict[str, Any]:
    data = load_json(EXAMPLES / "sample_data.json")
    matches = {item["match_id"]: match_from_dict(item) for item in data["matches"]}
    odds = {item["market_id"]: odds_from_dict(item) for item in data["odds"]}
    ratings = {item["entity_id"]: rating_from_dict(item) for item in data["ratings"]}
    styles = {item["entity_id"]: style_from_dict(item) for item in data["esports_styles"]}
    patch_meta_by_match = {item["match_id"]: patch_meta_from_dict(item["patch_meta"]) for item in data["patch_meta_by_match"]}
    football_context_by_match = {
        item["match_id"]: football_context_from_dict(item["football_context"]) for item in data["football_context_by_match"]
    }
    return {
        "matches": matches,
        "odds": odds,
        "ratings": ratings,
        "styles": styles,
        "patch_meta_by_match": patch_meta_by_match,
        "football_context_by_match": football_context_by_match,
        "historical_results": data["historical_results"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
