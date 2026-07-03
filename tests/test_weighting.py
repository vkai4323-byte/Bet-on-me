import unittest

from sportsbet_tool.modeling import HeuristicProbabilityModel
from sportsbet_tool.weighting import DEFAULT_FACTOR_WEIGHTS, FactorWeights, WeightUpdateConfig


class WeightingTests(unittest.TestCase):
    def test_weights_can_update_after_outcome(self):
        weights = FactorWeights(update_config=WeightUpdateConfig(learning_rate=0.1))
        before = weights.weights["h2h_delta"]
        weights.update_from_outcome({"h2h_delta": 1.0}, predicted_probability=0.55, won=True)
        self.assertGreater(weights.weights["h2h_delta"], before)

    def test_weight_updates_are_bounded(self):
        weights = FactorWeights(update_config=WeightUpdateConfig(learning_rate=0.5, max_multiplier=1.2))
        for _ in range(20):
            weights.update_from_outcome({"bp_fit_delta": 1.0}, predicted_probability=0.55, won=True)
        self.assertLessEqual(weights.weights["bp_fit_delta"], DEFAULT_FACTOR_WEIGHTS["bp_fit_delta"] * 1.2)

    def test_model_accepts_custom_weights(self):
        weights = FactorWeights()
        weights.weights["h2h_delta"] = 0.5
        model = HeuristicProbabilityModel(factor_weights=weights)
        self.assertIs(model.factor_weights, weights)


if __name__ == "__main__":
    unittest.main()
