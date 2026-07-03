from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sportsbet_tool.models import MarketOdds, Prediction, clean_dict
from sportsbet_tool.portfolio import PortfolioConfig, PortfolioPlanner
from sportsbet_tool.risk import BankrollStrategy, RiskMode, risk_config_for_mode


@dataclass(slots=True)
class ThinkingStep:
    name: str
    goal: str
    prompts: list[str]
    required_outputs: list[str]


@dataclass(slots=True)
class AIPlanningBrief:
    user_request: str
    bankroll: float
    risk_mode: RiskMode
    ai_role: str
    thinking_steps: list[ThinkingStep]
    decision_card_schema: dict[str, Any]
    safety_notes: list[str] = field(default_factory=list)


def build_research_brief(user_request: str, bankroll: float, risk_mode: RiskMode = "steady") -> AIPlanningBrief:
    """Create a protocol that guides AI research, prediction, and bet planning."""

    return AIPlanningBrief(
        user_request=user_request,
        bankroll=bankroll,
        risk_mode=risk_mode,
        ai_role=(
            "Act as an evidence-gathering analyst. The tool does not fetch live data or replace judgment; "
            "it standardizes what you collect, forces counterarguments, and turns your explicit probabilities "
            "into risk-controlled stake suggestions."
        ),
        thinking_steps=[
            ThinkingStep(
                name="scope",
                goal="Clarify exactly what the user wants to decide.",
                prompts=[
                    "Identify matches, sport, date/time, bankroll, risk appetite, and whether markets can include any selection.",
                    "Clarify market semantics when needed: 90-minute result, to-qualify, handicap, totals, team totals, or props.",
                ],
                required_outputs=["matches_to_analyze", "bankroll", "risk_mode", "allowed_market_types"],
            ),
            ThinkingStep(
                name="evidence_collection",
                goal="Collect source-backed facts before predicting.",
                prompts=[
                    "Gather fixtures, venue, stage, lineups/injuries/suspensions, recent form, tactical notes, rest/travel, and motivation.",
                    "Gather available odds for relevant markets and record bookmaker, decimal odds, timestamp, and source URL.",
                    "Separate confirmed facts, analyst opinion, market signal, and assumption.",
                ],
                required_outputs=["facts", "market_snapshot", "sources", "assumptions"],
            ),
            ThinkingStep(
                name="prediction_debate",
                goal="Make the AI argue with itself before producing probabilities.",
                prompts=[
                    "Write the base case for the predicted score and winner lean.",
                    "Write the strongest counter-case and what would make the pick fail.",
                    "Estimate probabilities conservatively and note uncertainty/disagreement.",
                ],
                required_outputs=["predicted_score", "winner_lean", "evidence_for", "evidence_against", "failure_modes"],
            ),
            ThinkingStep(
                name="market_translation",
                goal="Convert the match view into candidate bet markets.",
                prompts=[
                    "Choose markets that best express the prediction rather than forcing a fixed market type.",
                    "For each market, explain why it fits the score/winner view and why alternatives were rejected.",
                    "Only include a candidate when model probability exceeds fair implied probability by a clear edge.",
                ],
                required_outputs=["candidate_markets", "rejected_markets", "edge_rationale"],
            ),
            ThinkingStep(
                name="staking_plan",
                goal="Let the tool calculate stake sizes from explicit AI inputs.",
                prompts=[
                    "Provide model_probability, fair_implied_probability, quality_score, and disagreement for every candidate.",
                    "Lower quality_score when sources conflict, odds are stale, or the pick relies on uncertain news.",
                    "Increase disagreement when evidence and market signal conflict.",
                ],
                required_outputs=["model_probability", "quality_score", "disagreement", "source_url"],
            ),
        ],
        decision_card_schema=decision_card_schema(),
        safety_notes=[
            "Do not automate final bet submission.",
            "Treat the output as decision support, not a guarantee.",
            "The AI must show sources, assumptions, counterarguments, and failure modes before staking.",
            "If evidence is weak, the correct plan may be no bet.",
        ],
    )


def decision_card_schema() -> dict[str, Any]:
    return {
        "bankroll": "number",
        "risk_mode": "insurance|steady|adventurous|wild",
        "analyst_summary": {
            "thesis": "string",
            "main_risks": ["string"],
            "why_not_no_bet": "string",
        },
        "matches": [
            {
                "match_id": "string",
                "home": "string",
                "away": "string",
                "predicted_score": "string",
                "winner_lean": "string",
                "evidence_for": ["string"],
                "evidence_against": ["string"],
                "failure_modes": ["string"],
                "confidence_note": "string",
            }
        ],
        "markets": [
            {
                "match_id": "string",
                "market_id": "string",
                "market_type": "string",
                "selection": "string",
                "decimal_odds": "number",
                "fair_implied_probability": "number|null",
                "model_probability": "number",
                "quality_score": "number from 0 to 1",
                "disagreement": "number from 0 to 1",
                "bookmaker": "string",
                "source_url": "string",
                "why_this_market": "string",
                "failure_mode": "string",
                "rejected_alternatives": ["string"],
            }
        ],
        "sources": [{"title": "string", "url": "string", "notes": "string"}],
        "assumptions": ["string"],
    }


def plan_from_research_card(card: dict[str, Any]) -> dict[str, Any]:
    """Audit an AI decision card and convert its explicit probabilities into stake suggestions."""

    bankroll = float(card.get("bankroll", 1000.0))
    risk_mode = card.get("risk_mode", "steady")
    if risk_mode not in {"insurance", "steady", "adventurous", "wild"}:
        raise ValueError("risk_mode must be insurance, steady, adventurous, or wild")

    audit = audit_decision_card(card)
    strategy = BankrollStrategy(risk_config_for_mode(risk_mode))
    planner = PortfolioPlanner(
        PortfolioConfig(
            max_portfolio_fraction=strategy.config.max_daily_risk_fraction,
            max_match_fraction=min(strategy.config.max_single_bet_fraction, strategy.config.max_daily_risk_fraction),
            max_market_fraction=min(strategy.config.max_single_bet_fraction * 1.5, strategy.config.max_daily_risk_fraction),
        )
    )
    recommendations = []
    now = datetime.now(timezone.utc)
    for market in card.get("markets", []):
        probability = _bounded_probability(float(market["model_probability"]))
        quality_score = min(max(float(market.get("quality_score", 1.0)), 0.0), 1.0)
        disagreement = min(max(float(market.get("disagreement", 0.0)), 0.0), 1.0)
        if audit["severity"] == "weak":
            quality_score = min(quality_score, 0.5)
            disagreement = max(disagreement, 0.25)

        prediction = Prediction(
            match_id=market["match_id"],
            model_name="ai-decision-card-v0.1",
            target_side="home",
            probability=probability,
            confidence_low=max(0.0, probability - 0.12),
            confidence_high=min(1.0, probability + 0.12),
            factors={"ai_decision_card": 1.0},
            disagreement=disagreement,
            consensus_probability=probability,
        )
        odds = MarketOdds(
            match_id=market["match_id"],
            bookmaker=market.get("bookmaker", "manual-ai-research"),
            market_type=market["market_type"],
            side="home",
            decimal_odds=float(market["decimal_odds"]),
            observed_at=now,
            market_id=market["market_id"],
            metadata={
                "selection": market.get("selection"),
                "source_url": market.get("source_url"),
                "why_this_market": market.get("why_this_market"),
                "failure_mode": market.get("failure_mode"),
                "rejected_alternatives": market.get("rejected_alternatives", []),
            },
        )
        fair = market.get("fair_implied_probability")
        recommendation = strategy.recommend(
            prediction,
            odds,
            bankroll,
            fair_implied_probability=float(fair) if fair is not None else None,
            quality_score=quality_score,
            quality_reasons=["ai_decision_card_input", "manual_market_selection", *audit["quality_reasons"]],
        )
        recommendations.append(recommendation)

    planned = planner.plan(recommendations, bankroll)
    return {
        "bankroll": bankroll,
        "risk_mode": risk_mode,
        "analyst_summary": card.get("analyst_summary", {}),
        "thinking_audit": audit,
        "matches": card.get("matches", []),
        "recommendations": [clean_dict(item) for item in planned],
        "total_stake": round(sum(item.stake for item in planned if item.status == "recommended"), 2),
        "sources": card.get("sources", []),
        "assumptions": card.get("assumptions", []),
        "manual_confirmation_required": True,
    }


def audit_decision_card(card: dict[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    reasons: list[str] = []

    if not card.get("sources"):
        missing.append("sources")
        reasons.append("missing_sources")
    if not card.get("analyst_summary"):
        missing.append("analyst_summary")
        reasons.append("missing_analyst_summary")

    for index, match in enumerate(card.get("matches", [])):
        prefix = f"matches[{index}]"
        for key in ("evidence_for", "evidence_against", "failure_modes"):
            if not match.get(key):
                missing.append(f"{prefix}.{key}")
                reasons.append(f"missing_{key}")

    for index, market in enumerate(card.get("markets", [])):
        prefix = f"markets[{index}]"
        for key in ("why_this_market", "failure_mode", "source_url"):
            if not market.get(key):
                missing.append(f"{prefix}.{key}")
                reasons.append(f"missing_{key}")

    unique_reasons = _dedupe(reasons)
    if len(missing) >= 4:
        severity = "weak"
    elif missing:
        severity = "partial"
    else:
        severity = "complete"

    return {
        "severity": severity,
        "missing": missing,
        "quality_reasons": unique_reasons,
        "guidance": _audit_guidance(severity),
    }


def _audit_guidance(severity: str) -> str:
    if severity == "complete":
        return "AI card includes enough explicit evidence, counterarguments, and market rationale for stake planning."
    if severity == "partial":
        return "AI card is usable, but missing fields should lower confidence or be filled before real-money use."
    return "AI card is too thin; stake planning is heavily quality-penalized and no-bet should be considered."


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def _bounded_probability(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError("model_probability must be between 0 and 1")
    return min(max(value, 0.001), 0.999)
