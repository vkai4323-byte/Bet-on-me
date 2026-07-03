from __future__ import annotations

import argparse
import json
import sys
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
from sportsbet_tool.ensemble import ModelDebatePredictor
from sportsbet_tool.features import FeatureBuilder
from sportsbet_tool.intent import parse_intent
from sportsbet_tool.models import clean_dict
from sportsbet_tool.odds import no_vig_probabilities_by_market
from sportsbet_tool.portfolio import PortfolioConfig, PortfolioPlanner
from sportsbet_tool.quality import FeatureQualityAnalyzer
from sportsbet_tool.risk import BankrollStrategy, RiskMode, risk_config_for_mode

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
KNOWN_COMMANDS = {
    "demo-predict",
    "predict",
    "active",
    "all",
    "demo",
    "backtest",
    "bt",
    "ask",
    "自然语言",
    "prepare-slip",
    "slip",
}
GLOBAL_OPTIONS_WITH_VALUES = {"--config", "--bankroll", "--risk-mode"}


def main(argv: list[str] | None = None) -> int:
    argv = _normalize_argv(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Sports and esports prediction assistant.")
    parser.add_argument("--config", help="Optional JSON/YAML config path.")
    parser.add_argument("--bankroll", type=float, default=None, dest="global_bankroll")
    parser.add_argument("--risk-mode", choices=["insurance", "steady", "adventurous", "wild"], default=None, dest="global_risk_mode")
    parser.add_argument("--all", action="store_true", help="Print all evaluated bets instead of active recommendations.")
    sub = parser.add_subparsers(dest="command", required=False)

    demo = sub.add_parser(
        "demo-predict",
        aliases=["predict", "active", "all", "demo"],
        help="Run predictions on bundled sample data.",
    )
    demo.add_argument("--bankroll", type=float, default=None, dest="command_bankroll")
    demo.add_argument("--active-only", action="store_true", help="Only print recommended bets.")
    demo.add_argument("--risk-mode", choices=["insurance", "steady", "adventurous", "wild"], default=None, dest="command_risk_mode")

    backtest = sub.add_parser("backtest", aliases=["bt"], help="Run bundled historical sample backtest.")
    backtest.add_argument("--bankroll", type=float, default=None, dest="command_bankroll")
    backtest.add_argument("--risk-mode", choices=["insurance", "steady", "adventurous", "wild"], default=None, dest="command_risk_mode")

    ask = sub.add_parser("ask", aliases=["自然语言"], help="Infer settings from a natural-language request.")
    ask.add_argument("text", nargs="*")

    slip = sub.add_parser("prepare-slip", aliases=["slip"], help="Open sportsbook page and prepare manual bet-slip instructions.")
    slip.add_argument("--url", required=True)
    slip.add_argument("--stake-selector", default=None)
    slip.add_argument("--bankroll", type=float, default=None, dest="command_bankroll")
    slip.add_argument("--risk-mode", choices=["insurance", "steady", "adventurous", "wild"], default=None, dest="command_risk_mode")

    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command in {"ask", "自然语言"}:
        return _run_intent(" ".join(args.text), config, args.global_bankroll, args.global_risk_mode)

    risk_mode = _arg_value(args, "command_risk_mode", "global_risk_mode")
    if risk_mode:
        config.risk = risk_config_for_mode(risk_mode)
    bankroll = _arg_value(args, "command_bankroll", "global_bankroll")
    bankroll = bankroll if bankroll is not None else config.bankroll_amount

    if args.command is None or args.command in {"demo-predict", "predict", "active", "all", "demo"}:
        active_only = _default_active_only(args)
        return _demo_predict(bankroll, config, active_only=active_only)
    if args.command in {"backtest", "bt"}:
        return _backtest(bankroll, config)
    if args.command in {"prepare-slip", "slip"}:
        return _prepare_slip(args.url, args.stake_selector, bankroll, config)
    return 2


def _normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    intent_index = _first_command_or_intent_index(argv)
    if intent_index is None:
        return argv
    if argv[intent_index] in KNOWN_COMMANDS:
        return argv
    return [*argv[:intent_index], "ask", *argv[intent_index:]]


def _first_command_or_intent_index(argv: list[str]) -> int | None:
    index = 0
    while index < len(argv):
        item = argv[index]
        if item in GLOBAL_OPTIONS_WITH_VALUES:
            index += 2
            continue
        if item.startswith("-"):
            index += 1
            continue
        return index
    return None


def _run_intent(text: str, config: Any, bankroll_override: float | None = None, risk_mode_override: RiskMode | None = None) -> int:
    intent = parse_intent(text)
    if risk_mode_override:
        intent.risk_mode = risk_mode_override
        intent.reasons.append("risk_mode_override")
    config.risk = risk_config_for_mode(intent.risk_mode)
    bankroll = bankroll_override if bankroll_override is not None else intent.bankroll
    bankroll = bankroll if bankroll is not None else config.bankroll_amount

    if intent.command == "backtest":
        return _backtest(bankroll, config)
    return _demo_predict(bankroll, config, active_only=intent.active_only)


def _arg_value(args: argparse.Namespace, command_name: str, global_name: str) -> Any:
    command_value = getattr(args, command_name, None)
    return command_value if command_value is not None else getattr(args, global_name, None)


def _default_active_only(args: argparse.Namespace) -> bool:
    if getattr(args, "all", False) or args.command in {"all", "demo", "demo-predict"}:
        return bool(getattr(args, "active_only", False))
    return True


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
    result = BacktestEngine(
        model=ModelDebatePredictor(),
        strategy=BankrollStrategy(config.risk),
        planner=_build_portfolio_planner(config),
    ).run(samples, bankroll)
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
    model = ModelDebatePredictor()
    strategy = BankrollStrategy(config.risk)
    planner = _build_portfolio_planner(config)
    quality = FeatureQualityAnalyzer()
    recommendations = []
    day_candidates: list[Any] = []
    current_day: str | None = None
    default_odds = [odds for odds in bundle["odds"].values() if odds.bookmaker == config.default_bookmaker]
    fair_probabilities = no_vig_probabilities_by_market(default_odds)
    for odds in sorted(default_odds, key=lambda item: bundle["matches"][item.match_id].start_time):
        if odds.bookmaker != config.default_bookmaker:
            continue
        match = bundle["matches"][odds.match_id]
        match_day = match.start_time.date().isoformat()
        if match_day != current_day:
            if day_candidates:
                recommendations.extend(planner.plan(day_candidates, bankroll))
                day_candidates = []
            current_day = match_day
        vector = builder.build(
            match=match,
            odds=odds,
            player_ratings=bundle["ratings"],
            esports_styles=bundle["styles"],
            patch_meta=bundle["patch_meta_by_match"].get(match.match_id),
            football_context=bundle["football_context_by_match"].get(match.match_id),
        )
        quality_report = quality.score(vector)
        fair_probability = fair_probabilities.get(odds.market_id or "")
        prediction = model.predict_vector(vector, market_probability=fair_probability)
        recommendation = strategy.recommend(
            prediction,
            odds,
            bankroll,
            daily_risk_used=0.0,
            fair_implied_probability=fair_probability,
            quality_score=quality_report.score,
            quality_reasons=quality_report.reasons,
        )
        day_candidates.append(recommendation)
    if day_candidates:
        recommendations.extend(planner.plan(day_candidates, bankroll))
    return recommendations


def _build_portfolio_planner(config: Any) -> PortfolioPlanner:
    return PortfolioPlanner(
        PortfolioConfig(
            max_portfolio_fraction=config.risk.max_daily_risk_fraction,
            max_match_fraction=min(config.risk.max_single_bet_fraction, config.risk.max_daily_risk_fraction),
            max_market_fraction=min(config.risk.max_single_bet_fraction * 1.5, config.risk.max_daily_risk_fraction),
        )
    )


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
