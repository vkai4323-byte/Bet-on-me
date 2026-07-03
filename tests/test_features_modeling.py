import unittest

from sportsbet_tool.cli import _load_example_bundle
from sportsbet_tool.features import FeatureBuilder
from sportsbet_tool.modeling import HeuristicProbabilityModel


class FeatureModelingTests(unittest.TestCase):
    def test_esports_features_include_style_patch_and_h2h(self):
        bundle = _load_example_bundle()
        match = bundle["matches"]["lol-001"]
        odds = bundle["odds"]["bet365-lol-001-home-ml"]
        vector = FeatureBuilder().build(
            match,
            odds,
            player_ratings=bundle["ratings"],
            esports_styles=bundle["styles"],
            patch_meta=bundle["patch_meta_by_match"][match.match_id],
        )
        self.assertIn("elo_delta", vector.features)
        self.assertIn("patch_fit_delta", vector.features)
        self.assertIn("h2h_delta", vector.features)
        self.assertIn("pool_depth_delta", vector.features)

    def test_football_prematch_excludes_live_cards(self):
        bundle = _load_example_bundle()
        match = bundle["matches"]["football-001"]
        odds = bundle["odds"]["bet365-football-001-home-ml"]
        vector = FeatureBuilder().build(
            match,
            odds,
            player_ratings=bundle["ratings"],
            football_context=bundle["football_context_by_match"][match.match_id],
        )
        self.assertIn("referee_yellow_rate", vector.features)
        self.assertIn("motivation_delta", vector.features)
        self.assertNotIn("live_red_card_delta", vector.features)

    def test_model_probability_is_bounded(self):
        bundle = _load_example_bundle()
        match = bundle["matches"]["football-001"]
        odds = bundle["odds"]["bet365-football-001-home-ml"]
        vector = FeatureBuilder().build(
            match,
            odds,
            player_ratings=bundle["ratings"],
            football_context=bundle["football_context_by_match"][match.match_id],
        )
        prediction = HeuristicProbabilityModel().predict_vector(vector)
        self.assertGreater(prediction.probability, 0.03)
        self.assertLess(prediction.probability, 0.97)
        self.assertLessEqual(prediction.confidence_low, prediction.probability)
        self.assertGreaterEqual(prediction.confidence_high, prediction.probability)


if __name__ == "__main__":
    unittest.main()
