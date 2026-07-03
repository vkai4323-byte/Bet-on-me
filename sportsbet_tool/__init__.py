"""Sports and esports prediction and safe bet-slip assistance toolkit."""

from sportsbet_tool.models import (
    BetRecommendation,
    BetRecord,
    EsportsStyleSnapshot,
    FootballContextSnapshot,
    MarketOdds,
    Match,
    PatchMetaSnapshot,
    PlayerRatingSnapshot,
    Prediction,
)
from sportsbet_tool.ensemble import ModelDebatePredictor
from sportsbet_tool.intent import IntentSettings, parse_intent
from sportsbet_tool.portfolio import PortfolioPlanner
from sportsbet_tool.research import audit_decision_card, build_research_brief, plan_from_research_card

__all__ = [
    "BetRecommendation",
    "BetRecord",
    "EsportsStyleSnapshot",
    "FootballContextSnapshot",
    "MarketOdds",
    "Match",
    "ModelDebatePredictor",
    "PatchMetaSnapshot",
    "PlayerRatingSnapshot",
    "PortfolioPlanner",
    "Prediction",
    "IntentSettings",
    "audit_decision_card",
    "build_research_brief",
    "plan_from_research_card",
    "parse_intent",
]
