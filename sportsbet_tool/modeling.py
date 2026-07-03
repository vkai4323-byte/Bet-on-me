from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

from sportsbet_tool.features import FeatureVector
from sportsbet_tool.models import Prediction


def sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def logit(value: float) -> float:
    clipped = min(max(value, 1e-6), 1.0 - 1e-6)
    return log(clipped / (1.0 - clipped))


def calibrate_probability(probability: float, temperature: float = 1.15, floor: float = 0.03) -> float:
    calibrated = sigmoid(logit(probability) / temperature)
    return min(max(calibrated, floor), 1.0 - floor)


@dataclass(slots=True)
class ModelOutput:
    probability: float
    factors: dict[str, float]


class HeuristicProbabilityModel:
    """A deterministic baseline model that can be replaced by fitted ML models."""

    model_name = "heuristic-v0.1"

    def __init__(self, temperature: float = 1.15) -> None:
        self.temperature = temperature
        self.weights = {
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

    def predict_vector(self, vector: FeatureVector) -> Prediction:
        output = self.score(vector)
        confidence = self._confidence_width(vector.features)
        return Prediction(
            match_id=vector.match_id,
            model_name=self.model_name,
            target_side=vector.target_side,  # type: ignore[arg-type]
            probability=output.probability,
            confidence_low=max(0.0, output.probability - confidence),
            confidence_high=min(1.0, output.probability + confidence),
            factors=output.factors,
        )

    def score(self, vector: FeatureVector) -> ModelOutput:
        intercept = 0.0
        if vector.features.get("is_home_side", 0.0) > 0.5 and vector.features.get("neutral_site", 0.0) < 0.5:
            intercept += 0.13
        contributions: dict[str, float] = {}
        logit_score = intercept
        if intercept:
            contributions["home_field_intercept"] = intercept
        for key, weight in self.weights.items():
            if key not in vector.features:
                continue
            contribution = vector.features[key] * weight
            if contribution:
                contributions[key] = contribution
                logit_score += contribution
        probability = calibrate_probability(sigmoid(logit_score), temperature=self.temperature)
        return ModelOutput(probability=probability, factors=contributions)

    @staticmethod
    def _confidence_width(features: dict[str, float]) -> float:
        sample_bonus = min(abs(features.get("rating_sample_delta", 0.0)) * 0.002, 0.03)
        data_weight = min(max(features.get("patch_data_weight", 1.0), 0.0), 1.0)
        base = 0.12 - sample_bonus
        if data_weight < 0.7:
            base += (0.7 - data_weight) * 0.12
        return min(max(base, 0.05), 0.22)
