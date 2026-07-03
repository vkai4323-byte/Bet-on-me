from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

from sportsbet_tool.features import FeatureVector
from sportsbet_tool.models import Prediction
from sportsbet_tool.weighting import FactorWeights


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

    def __init__(self, temperature: float = 1.15, factor_weights: FactorWeights | None = None) -> None:
        self.temperature = temperature
        self.factor_weights = factor_weights or FactorWeights()

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
        for key, value in vector.features.items():
            weight = self.factor_weights.get(key)
            if weight is None:
                continue
            contribution = value * weight
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
