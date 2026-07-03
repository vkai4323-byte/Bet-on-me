from __future__ import annotations

from dataclasses import dataclass, replace

from sportsbet_tool.models import BetRecommendation


@dataclass(slots=True)
class PortfolioConfig:
    max_portfolio_fraction: float = 0.05
    max_match_fraction: float = 0.018
    max_market_fraction: float = 0.025
    disagreement_stake_penalty: float = 0.75
    same_match_keep_best_only: bool = True

    def validate(self) -> None:
        if not 0.0 < self.max_portfolio_fraction <= 1.0:
            raise ValueError("max_portfolio_fraction must be in (0, 1]")
        if not 0.0 < self.max_match_fraction <= 1.0:
            raise ValueError("max_match_fraction must be in (0, 1]")
        if not 0.0 < self.max_market_fraction <= 1.0:
            raise ValueError("max_market_fraction must be in (0, 1]")
        if not 0.0 <= self.disagreement_stake_penalty <= 1.0:
            raise ValueError("disagreement_stake_penalty must be between 0 and 1")


class PortfolioPlanner:
    """Allocates bankroll across candidate bets after single-bet risk checks."""

    def __init__(self, config: PortfolioConfig | None = None) -> None:
        self.config = config or PortfolioConfig()
        self.config.validate()

    def plan(self, recommendations: list[BetRecommendation], bankroll: float) -> list[BetRecommendation]:
        if bankroll <= 0:
            raise ValueError("bankroll must be positive")

        planned = [self._skip_non_positive_edge(item) for item in recommendations]
        planned = self._apply_same_match_filter(planned)

        budget_remaining = bankroll * self.config.max_portfolio_fraction
        match_exposure: dict[str, float] = {}
        market_exposure: dict[str, float] = {}
        output: list[BetRecommendation] = []

        for item in sorted(planned, key=self._priority, reverse=True):
            if item.status != "recommended" or item.stake <= 0:
                output.append(item)
                continue

            reasons = list(item.reasons)
            adjusted_stake = self._stake_after_disagreement(item)
            if adjusted_stake < item.stake:
                reasons.append("portfolio_disagreement_scaled_stake")

            match_remaining = bankroll * self.config.max_match_fraction - match_exposure.get(item.match_id, 0.0)
            market_key = f"{item.match_id}:{item.market_type}"
            market_remaining = bankroll * self.config.max_market_fraction - market_exposure.get(market_key, 0.0)
            capped_stake = max(0.0, min(adjusted_stake, budget_remaining, match_remaining, market_remaining))

            status = item.status
            if capped_stake < adjusted_stake:
                reasons.append("portfolio_exposure_cap")
            if capped_stake <= 0:
                reasons.append("portfolio_budget_exhausted")
                status = "skipped"

            rounded_stake = round(capped_stake, 2)
            planned_item = replace(
                item,
                stake=rounded_stake,
                applied_fraction=rounded_stake / bankroll,
                status=status,
                reasons=self._dedupe(reasons),
            )
            output.append(planned_item)
            if status == "recommended" and rounded_stake > 0:
                budget_remaining -= rounded_stake
                match_exposure[item.match_id] = match_exposure.get(item.match_id, 0.0) + rounded_stake
                market_exposure[market_key] = market_exposure.get(market_key, 0.0) + rounded_stake

        order = {self._identity(item): index for index, item in enumerate(recommendations)}
        return sorted(output, key=lambda item: order.get(self._identity(item), len(order)))

    def _apply_same_match_filter(self, recommendations: list[BetRecommendation]) -> list[BetRecommendation]:
        if not self.config.same_match_keep_best_only:
            return recommendations

        best_by_match: dict[str, BetRecommendation] = {}
        for item in recommendations:
            if item.status != "recommended" or item.stake <= 0:
                continue
            current = best_by_match.get(item.match_id)
            if current is None or self._priority(item) > self._priority(current):
                best_by_match[item.match_id] = item

        best_ids = {self._identity(item) for item in best_by_match.values()}
        filtered: list[BetRecommendation] = []
        for item in recommendations:
            if item.status == "recommended" and item.stake > 0 and self._identity(item) not in best_ids:
                filtered.append(
                    replace(
                        item,
                        stake=0.0,
                        applied_fraction=0.0,
                        status="skipped",
                        reasons=self._dedupe([*item.reasons, "portfolio_same_match_lower_rank"]),
                    )
                )
            else:
                filtered.append(item)
        return filtered

    def _stake_after_disagreement(self, item: BetRecommendation) -> float:
        if item.model_disagreement <= 0:
            return item.stake
        multiplier = max(0.0, 1.0 - self.config.disagreement_stake_penalty * item.model_disagreement)
        return item.stake * multiplier

    @staticmethod
    def _skip_non_positive_edge(item: BetRecommendation) -> BetRecommendation:
        if item.status == "recommended" and item.edge <= 0:
            return replace(
                item,
                stake=0.0,
                applied_fraction=0.0,
                status="skipped",
                reasons=PortfolioPlanner._dedupe([*item.reasons, "portfolio_non_positive_edge"]),
            )
        return item

    @staticmethod
    def _priority(item: BetRecommendation) -> float:
        quality = max(item.quality_score, 0.05)
        disagreement_penalty = max(0.1, 1.0 - item.model_disagreement)
        return item.expected_value * quality * disagreement_penalty

    @staticmethod
    def _identity(item: BetRecommendation) -> tuple[str, str | None, str, str]:
        return (item.match_id, item.market_id, item.market_type, item.side)

    @staticmethod
    def _dedupe(reasons: list[str]) -> list[str]:
        output: list[str] = []
        for reason in reasons:
            if reason not in output:
                output.append(reason)
        return output
