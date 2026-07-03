from __future__ import annotations

from dataclasses import dataclass

from sportsbet_tool.models import BetRecommendation, MarketOdds, Prediction
from sportsbet_tool.odds import decimal_kelly_fraction, expected_value, implied_probability


@dataclass(slots=True)
class RiskConfig:
    kelly_fraction: float = 0.5
    max_single_bet_fraction: float = 0.02
    max_daily_risk_fraction: float = 0.05
    min_edge: float = 0.015
    pause_after_consecutive_losses: int = 5
    min_stake: float = 0.0

    def validate(self) -> None:
        if not 0.0 < self.kelly_fraction <= 1.0:
            raise ValueError("kelly_fraction must be in (0, 1]")
        if not 0.0 < self.max_single_bet_fraction <= 1.0:
            raise ValueError("max_single_bet_fraction must be in (0, 1]")
        if not 0.0 < self.max_daily_risk_fraction <= 1.0:
            raise ValueError("max_daily_risk_fraction must be in (0, 1]")
        if self.min_edge < 0:
            raise ValueError("min_edge cannot be negative")


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
    ) -> BetRecommendation:
        if bankroll <= 0:
            raise ValueError("bankroll must be positive")
        if prediction.match_id != odds.match_id:
            raise ValueError("prediction and odds must reference the same match")
        if prediction.target_side != odds.side:
            raise ValueError("prediction target side and odds side must match")

        imp = implied_probability(odds.decimal_odds)
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

        single_cap = bankroll * self.config.max_single_bet_fraction
        daily_cap_remaining = max(0.0, bankroll * self.config.max_daily_risk_fraction - daily_risk_used)
        uncapped_stake = bankroll * applied_fraction
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
