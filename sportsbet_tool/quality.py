from __future__ import annotations

from dataclasses import dataclass, field

from sportsbet_tool.features import FeatureVector


@dataclass(slots=True)
class FeatureQualityReport:
    score: float
    reasons: list[str] = field(default_factory=list)


class FeatureQualityAnalyzer:
    expected_by_sport = {
        "football": {
            "elo_delta",
            "referee_yellow_rate",
            "motivation_delta",
            "injury_impact_delta",
            "weather_disruption",
        },
        "lol": {"elo_delta", "patch_fit_delta", "h2h_delta", "bp_fit_delta", "pool_depth_delta"},
        "valorant": {"elo_delta", "patch_fit_delta", "h2h_delta", "bp_fit_delta", "map_pool_delta"},
        "cs2": {"elo_delta", "patch_fit_delta", "h2h_delta", "bp_fit_delta", "map_pool_delta"},
    }

    def score(self, vector: FeatureVector) -> FeatureQualityReport:
        features = vector.features
        score = 1.0
        reasons: list[str] = []

        expected = self.expected_by_sport.get(vector.sport, set())
        missing = sorted(key for key in expected if key not in features)
        if missing:
            score -= min(0.45, 0.07 * len(missing))
            reasons.append("missing_core_factors:" + ",".join(missing))

        rating_min_sample = features.get("rating_min_sample")
        if rating_min_sample is not None and rating_min_sample < 8:
            score -= 0.15
            reasons.append("low_rating_sample")

        style_min_sample = features.get("style_min_sample")
        if style_min_sample is not None and style_min_sample < 10:
            score -= 0.12
            reasons.append("low_style_sample")

        patch_weight = features.get("patch_data_weight")
        if patch_weight is not None and patch_weight < 0.7:
            score -= (0.7 - patch_weight) * 0.35
            reasons.append("low_patch_data_weight")

        return FeatureQualityReport(score=min(max(score, 0.0), 1.0), reasons=reasons)
