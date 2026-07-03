from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from sportsbet_tool.features import FeatureVector
from sportsbet_tool.modeling import HeuristicProbabilityModel, logit, sigmoid
from sportsbet_tool.models import Prediction
from sportsbet_tool.weighting import DEFAULT_FACTOR_WEIGHTS, FactorWeights


@dataclass(slots=True)
class ModelOpinion:
    name: str
    probability: float
    weight: float = 1.0


class FeatureSubsetProbabilityModel(HeuristicProbabilityModel):
    def __init__(self, model_name: str, feature_names: set[str], temperature: float = 1.15) -> None:
        weights = {key: value for key, value in DEFAULT_FACTOR_WEIGHTS.items() if key in feature_names}
        super().__init__(temperature=temperature, factor_weights=FactorWeights(weights=weights))
        self.model_name = model_name


class ModelDebatePredictor:
    """Combines independent model views and penalizes fragile consensus."""

    model_name = "debate-ensemble-v0.1"

    disagreement_veto_threshold = 0.18
    market_veto_threshold = 0.14

    def __init__(self, models: list[HeuristicProbabilityModel] | None = None) -> None:
        self.models = models or [
            HeuristicProbabilityModel(),
            FeatureSubsetProbabilityModel(
                "strength-only-v0.1",
                {
                    "elo_delta",
                    "elo_7d_delta",
                    "elo_30d_delta",
                    "opponent_strength_delta",
                    "rating_sample_delta",
                },
                temperature=1.25,
            ),
            FeatureSubsetProbabilityModel(
                "context-shock-v0.1",
                {
                    "patch_fit_delta",
                    "patch_pool_overlap_delta",
                    "patch_meta_shift",
                    "motivation_delta",
                    "schedule_pressure_delta",
                    "injury_impact_delta",
                    "suspension_impact_delta",
                    "referee_home_bias",
                    "weather_disruption",
                    "venue_home_familiarity",
                    "card_tendency_delta",
                    "set_piece_delta",
                },
                temperature=1.35,
            ),
            FeatureSubsetProbabilityModel(
                "style-tactics-v0.1",
                {
                    "style_aggression_delta",
                    "style_first_action_delta",
                    "style_clutch_delta",
                    "style_info_delta",
                    "style_teamfight_delta",
                    "style_consistency_delta",
                    "pool_depth_delta",
                    "meta_mastery_delta",
                    "map_pool_delta",
                    "h2h_delta",
                    "bp_fit_delta",
                    "synergy_delta",
                    "style_clash_home_advantage",
                },
                temperature=1.3,
            ),
        ]

    def predict_vector(self, vector: FeatureVector, market_probability: float | None = None) -> Prediction:
        opinions = [ModelOpinion(model.model_name, model.predict_vector(vector).probability) for model in self.models]
        weighted_logits = [logit(item.probability) * item.weight for item in opinions]
        total_weight = sum(item.weight for item in opinions)
        consensus = sigmoid(sum(weighted_logits) / total_weight)
        disagreement = max(item.probability for item in opinions) - min(item.probability for item in opinions)
        average_distance = mean(abs(item.probability - consensus) for item in opinions)
        conservative = self._conservative_probability(consensus, disagreement, average_distance)
        veto_reasons = self._veto_reasons(opinions, consensus, disagreement, market_probability)
        confidence = self._confidence_width(disagreement, average_distance, bool(veto_reasons))

        return Prediction(
            match_id=vector.match_id,
            model_name=self.model_name,
            target_side=vector.target_side,  # type: ignore[arg-type]
            probability=conservative,
            confidence_low=max(0.0, conservative - confidence),
            confidence_high=min(1.0, conservative + confidence),
            factors={
                "consensus_probability": consensus,
                "model_disagreement": disagreement,
                "average_model_distance": average_distance,
                "conservative_adjustment": consensus - conservative,
            },
            model_probabilities=self._probability_map(opinions, market_probability),
            disagreement=disagreement,
            veto_reasons=veto_reasons,
            consensus_probability=consensus,
            market_probability=market_probability,
        )

    @staticmethod
    def _probability_map(opinions: list[ModelOpinion], market_probability: float | None) -> dict[str, float]:
        probabilities = {item.name: item.probability for item in opinions}
        if market_probability is not None:
            probabilities["market-no-vig"] = market_probability
        return probabilities

    @staticmethod
    def _conservative_probability(consensus: float, disagreement: float, average_distance: float) -> float:
        penalty = 0.35 * disagreement + 0.5 * average_distance
        adjusted = consensus - penalty
        return min(max(adjusted, 0.03), 0.97)

    def _veto_reasons(
        self,
        opinions: list[ModelOpinion],
        consensus: float,
        disagreement: float,
        market_probability: float | None,
    ) -> list[str]:
        reasons: list[str] = []
        if disagreement >= self.disagreement_veto_threshold:
            reasons.append("high_model_disagreement")
        if market_probability is not None and abs(consensus - market_probability) >= self.market_veto_threshold:
            reasons.append("large_market_model_gap")
        same_side_votes = sum(1 for item in opinions if (item.probability - 0.5) * (consensus - 0.5) > 0)
        if same_side_votes < max(2, len(opinions) - 1):
            reasons.append("weak_model_consensus")
        return reasons

    @staticmethod
    def _confidence_width(disagreement: float, average_distance: float, has_veto: bool) -> float:
        width = 0.08 + 0.45 * disagreement + 0.5 * average_distance
        if has_veto:
            width += 0.04
        return min(max(width, 0.07), 0.28)
