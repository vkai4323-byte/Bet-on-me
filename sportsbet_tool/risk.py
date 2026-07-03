from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sportsbet_tool.models import BetRecommendation, MarketOdds, Prediction
from sportsbet_tool.odds import decimal_kelly_fraction, expected_value, implied_probability

RiskMode = Literal["insurance", "steady", "adventurous", "wild"]


@dataclass(slots=True)
class RiskConfig:
    kelly_fraction: float = 0.5
    max_single_bet_fraction: float = 0.02
    max_daily_risk_fraction: float = 0.05
    min_edge: float = 0.015
    pause_after_consecutive_losses: int = 5
    min_stake: float = 0.0
    min_quality_score: float = 0.55
    mode: RiskMode = "steady"

    def validate(self) -> None:
        if not 0.0 < self.kelly_fraction <= 1.0:
            raise ValueError("kelly_fraction must be in (0, 1]")
        if not 0.0 < self.max_single_bet_fraction <= 1.0:
            raise ValueError("max_single_bet_fraction must be in (0, 1]")
        if not 0.0 < self.max_daily_risk_fraction <= 1.0:
            raise ValueError("max_daily_risk_fraction must be in (0, 1]")
        if self.min_edge < 0:
            raise ValueError("min_edge cannot be negative")
        if not 0.0 <= self.min_quality_score <= 1.0:
            raise ValueError("min_quality_score must be between 0 and 1")


MODE_CONFIGS: dict[RiskMode, RiskConfig] = {
    "insurance": RiskConfig(
        kelly_fraction=0.2,
        max_single_bet_fraction=0.006,
        max_daily_risk_fraction=0.018,
        min_edge=0.04,
        pause_after_consecutive_losses=2,
        min_quality_score=0.78,
        mode="insurance",
    ),
    "steady": RiskConfig(
        kelly_fraction=0.5,
        max_single_bet_fraction=0.02,
        max_daily_risk_fraction=0.05,
        min_edge=0.015,
        pause_after_consecutive_losses=5,
        min_quality_score=0.55,
        mode="steady",
    ),
    "adventurous": RiskConfig(
        kelly_fraction=0.75,
        max_single_bet_fraction=0.035,
        max_daily_risk_fraction=0.09,
        min_edge=0.008,
        pause_after_consecutive_losses=7,
        min_quality_score=0.45,
        mode="adventurous",
    ),
    "wild": RiskConfig(
        kelly_fraction=1.0,
        max_single_bet_fraction=0.06,
        max_daily_risk_fraction=0.16,
        min_edge=0.0,
        pause_after_consecutive_losses=10,
        min_quality_score=0.35,
        mode="wild",
    ),
}


def risk_config_for_mode(mode: RiskMode) -> RiskConfig:
    config = MODE_CONFIGS[mode]
    return RiskConfig(
        kelly_fraction=config.kelly_fraction,
        max_single_bet_fraction=config.max_single_bet_fraction,
        max_daily_risk_fraction=config.max_daily_risk_fraction,
        min_edge=config.min_edge,
        pause_after_consecutive_losses=config.pause_after_consecutive_losses,
        min_quality_score=config.min_quality_score,
        mode=config.mode,
    )


class BankrollStrategy:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self.config.validate()

    def recommend(
        self,
        prediction: Prediction,
        odds: MarketOdds,
        bankroll: float,
        daily_risk_used: float = 0.0,
        consecutive_losses: int = 0,
        fair_implied_probability: float | None = None,
        quality_score: float = 1.0,
        quality_reasons: list[str] | None = None,
    ) -> BetRecommendation:
        if bankroll <= 0:
            raise ValueError("bankroll must be positive")
        if prediction.match_id != odds.match_id:
            raise ValueError("prediction and odds must reference the same match")
        if prediction.target_side != odds.side:
            raise ValueError("prediction target side and odds side must match")

        if fair_implied_probability is not None and not 0.0 < fair_implied_probability < 1.0:
            raise ValueError("fair_implied_probability must be in (0, 1)")
        quality_score = min(max(quality_score, 0.0), 1.0)

        book_imp = implied_probability(odds.decimal_odds)
        imp = fair_implied_probability if fair_implied_probability is not None else book_imp
        edge = prediction.probability - imp
        ev = expected_value(prediction.probability, odds.decimal_odds)
        raw_kelly = decimal_kelly_fraction(prediction.probability, odds.decimal_odds)
        applied_fraction = raw_kelly * self.config.kelly_fraction
        reasons: list[str] = []
        status = "recommended"

        if consecutive_losses >= self.config.pause_after_consecutive_losses:
            reasons.append("paused_after_consecutive_losses")
            status = "paused"
            applied_fraction = 0.0

        if edge < self.config.min_edge:
            reasons.append("edge_below_threshold")
            status = "skipped"
            applied_fraction = 0.0

        if ev <= 0:
            reasons.append("non_positive_expected_value")
            status = "skipped"
            applied_fraction = 0.0

        if fair_implied_probability is not None:
            reasons.append("no_vig_implied_probability")

        if quality_score < self.config.min_quality_score:
            reasons.append("quality_below_threshold")
            status = "skipped"
            applied_fraction = 0.0
        elif quality_score < 0.95:
            reasons.append("quality_scaled_stake")

        for reason in quality_reasons or []:
            if reason not in reasons:
                reasons.append(reason)

        single_cap = bankroll * self.config.max_single_bet_fraction
        daily_cap_remaining = max(0.0, bankroll * self.config.max_daily_risk_fraction - daily_risk_used)
        uncapped_stake = bankroll * applied_fraction * quality_score
        stake = max(0.0, min(uncapped_stake, single_cap, daily_cap_remaining))

        if daily_cap_remaining <= 0 and status == "recommended":
            reasons.append("daily_risk_cap_reached")
            status = "skipped"
            stake = 0.0

        if self.config.min_stake > 0 and stake < self.config.min_stake and status == "recommended":
            reasons.append("stake_below_minimum")
            status = "skipped"
            stake = 0.0

        risk_label = self._risk_label(edge=edge, stake_fraction=stake / bankroll)

        return BetRecommendation(
            match_id=odds.match_id,
            market_id=odds.market_id,
            bookmaker=odds.bookmaker,
            market_type=odds.market_type,
            side=odds.side,
            decimal_odds=odds.decimal_odds,
            model_probability=prediction.probability,
            implied_probability=imp,
            edge=edge,
            expected_value=ev,
            raw_kelly_fraction=raw_kelly,
            applied_fraction=stake / bankroll if bankroll else 0.0,
            stake=round(stake, 2),
            risk_label=risk_label,
            status=status,
            reasons=reasons,
            requires_manual_confirmation=True,
            book_implied_probability=book_imp,
            quality_score=quality_score,
        )

    @staticmethod
    def _risk_label(edge: float, stake_fraction: float) -> str:
        if stake_fraction <= 0:
            return "none"
        if edge >= 0.08 and stake_fraction >= 0.015:
            return "elevated"
        if edge >= 0.035:
            return "standard"
        return "thin_edge"
