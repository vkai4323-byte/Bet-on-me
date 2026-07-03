from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sportsbet_tool.features import FeatureVector
from sportsbet_tool.modeling import HeuristicProbabilityModel
from sportsbet_tool.models import BetRecommendation, MarketOdds
from sportsbet_tool.risk import BankrollStrategy, RiskConfig


@dataclass(slots=True)
class BacktestSample:
    start_time: datetime
    vector: FeatureVector
    odds: MarketOdds
    actual_side: str
    closing_decimal_odds: float | None = None


@dataclass(slots=True)
class BacktestResult:
    starting_bankroll: float
    ending_bankroll: float
    total_staked: float
    total_profit: float
    roi: float
    bets: int
    wins: int
    hit_rate: float
    max_drawdown: float
    average_clv: float | None
    calibration_buckets: dict[str, dict[str, float]]
    recommendations: list[BetRecommendation] = field(default_factory=list)


class BacktestEngine:
    def __init__(
        self,
        model: HeuristicProbabilityModel | None = None,
        strategy: BankrollStrategy | None = None,
    ) -> None:
        self.model = model or HeuristicProbabilityModel()
        self.strategy = strategy or BankrollStrategy(RiskConfig())

    def run(self, samples: list[BacktestSample], starting_bankroll: float) -> BacktestResult:
        bankroll = starting_bankroll
        peak = starting_bankroll
        max_drawdown = 0.0
        total_staked = 0.0
        total_profit = 0.0
        bets = 0
        wins = 0
        daily_risk_used = 0.0
        current_day: str | None = None
        consecutive_losses = 0
        clv_values: list[float] = []
        buckets: dict[str, list[tuple[float, int]]] = {}
        recommendations: list[BetRecommendation] = []

        for sample in sorted(samples, key=lambda item: item.start_time):
            day = sample.start_time.date().isoformat()
            if day != current_day:
                current_day = day
                daily_risk_used = 0.0

            prediction = self.model.predict_vector(sample.vector)
            recommendation = self.strategy.recommend(
                prediction,
                sample.odds,
                bankroll,
                daily_risk_used=daily_risk_used,
                consecutive_losses=consecutive_losses,
            )
            recommendations.append(recommendation)
            self._add_calibration_sample(buckets, prediction.probability, sample.actual_side == sample.odds.side)

            if recommendation.status != "recommended" or recommendation.stake <= 0:
                continue

            bets += 1
            total_staked += recommendation.stake
            daily_risk_used += recommendation.stake
            if sample.actual_side == sample.odds.side:
                profit = recommendation.stake * (sample.odds.decimal_odds - 1.0)
                wins += 1
                consecutive_losses = 0
            else:
                profit = -recommendation.stake
                consecutive_losses += 1

            bankroll += profit
            total_profit += profit
            peak = max(peak, bankroll)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - bankroll) / peak)

            if sample.closing_decimal_odds:
                clv_values.append((1.0 / sample.closing_decimal_odds) - (1.0 / sample.odds.decimal_odds))

        roi = total_profit / total_staked if total_staked else 0.0
        hit_rate = wins / bets if bets else 0.0
        average_clv = sum(clv_values) / len(clv_values) if clv_values else None
        return BacktestResult(
            starting_bankroll=starting_bankroll,
            ending_bankroll=round(bankroll, 2),
            total_staked=round(total_staked, 2),
            total_profit=round(total_profit, 2),
            roi=roi,
            bets=bets,
            wins=wins,
            hit_rate=hit_rate,
            max_drawdown=max_drawdown,
            average_clv=average_clv,
            calibration_buckets=self._summarize_buckets(buckets),
            recommendations=recommendations,
        )

    @staticmethod
    def _add_calibration_sample(buckets: dict[str, list[tuple[float, int]]], probability: float, won: bool) -> None:
        low = int(probability * 10) / 10
        high = low + 0.1
        key = f"{low:.1f}-{high:.1f}"
        buckets.setdefault(key, []).append((probability, 1 if won else 0))

    @staticmethod
    def _summarize_buckets(buckets: dict[str, list[tuple[float, int]]]) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for key, values in buckets.items():
            count = len(values)
            summary[key] = {
                "count": float(count),
                "avg_probability": sum(item[0] for item in values) / count,
                "actual_win_rate": sum(item[1] for item in values) / count,
            }
        return summary
