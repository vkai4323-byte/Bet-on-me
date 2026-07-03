from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_FACTOR_WEIGHTS = {
    "elo_delta": 0.0026,
    "elo_7d_delta": 0.0008,
    "elo_30d_delta": 0.0009,
    "opponent_strength_delta": 0.004,
    "rating_sample_delta": 0.006,
    "patch_fit_delta": 0.34,
    "patch_pool_overlap_delta": 0.22,
    "patch_meta_shift": -0.05,
    "style_aggression_delta": 0.08,
    "style_first_action_delta": 0.11,
    "style_clutch_delta": 0.16,
    "style_info_delta": 0.12,
    "style_teamfight_delta": 0.11,
    "style_consistency_delta": 0.14,
    "pool_depth_delta": 0.12,
    "meta_mastery_delta": 0.18,
    "map_pool_delta": 0.09,
    "h2h_delta": 0.16,
    "bp_fit_delta": 0.16,
    "synergy_delta": 0.18,
    "style_clash_home_advantage": 0.15,
    "motivation_delta": 0.20,
    "schedule_pressure_delta": 0.10,
    "injury_impact_delta": 0.16,
    "suspension_impact_delta": 0.14,
    "referee_home_bias": 0.08,
    "weather_disruption": -0.04,
    "venue_home_familiarity": 0.11,
    "card_tendency_delta": 0.07,
    "set_piece_delta": 0.10,
    "live_red_card_delta": 0.45,
    "live_yellow_card_delta": 0.06,
}


@dataclass(slots=True)
class WeightUpdateConfig:
    learning_rate: float = 0.03
    min_multiplier: float = 0.35
    max_multiplier: float = 2.5
    min_abs_weight: float = 0.0001


@dataclass(slots=True)
class FactorWeights:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FACTOR_WEIGHTS))
    update_config: WeightUpdateConfig = field(default_factory=WeightUpdateConfig)

    def get(self, feature_name: str) -> float | None:
        return self.weights.get(feature_name)

    def update_from_outcome(
        self,
        features: dict[str, float],
        predicted_probability: float,
        won: bool,
    ) -> dict[str, float]:
        error = (1.0 if won else 0.0) - predicted_probability
        updated: dict[str, float] = {}
        for feature_name, value in features.items():
            if feature_name not in self.weights or value == 0:
                continue
            current = self.weights[feature_name]
            direction = 1.0 if current * value * error > 0 else -1.0
            candidate = current * (1.0 + direction * self.update_config.learning_rate)
            bounded = self._bounded(feature_name, candidate)
            self.weights[feature_name] = bounded
            updated[feature_name] = bounded
        return updated

    def _bounded(self, feature_name: str, value: float) -> float:
        base = DEFAULT_FACTOR_WEIGHTS.get(feature_name, value)
        low = min(base * self.update_config.min_multiplier, base * self.update_config.max_multiplier)
        high = max(base * self.update_config.min_multiplier, base * self.update_config.max_multiplier)
        if abs(base) < self.update_config.min_abs_weight:
            return value
        return min(max(value, low), high)
