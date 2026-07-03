from __future__ import annotations

import unittest
from datetime import datetime, timezone

from sportsbet_tool.cli import _load_example_bundle
from sportsbet_tool.ensemble import ModelDebatePredictor
from sportsbet_tool.features import FeatureBuilder
from sportsbet_tool.models import BetRecommendation, MarketOdds, Prediction
from sportsbet_tool.odds import no_vig_probabilities_by_market
from sportsbet_tool.portfolio import PortfolioConfig, PortfolioPlanner
from sportsbet_tool.risk import BankrollStrategy, RiskConfig


class EnsemblePortfolioTests(unittest.TestCase):
    def test_debate_prediction_records_model_disagreement_and_market_view(self):
        bundle = _load_example_bundle()
        odds = bundle["odds"]["bet365-lol-001-home-ml"]
        match = bundle["matches"][odds.match_id]
        vector = FeatureBuilder().build(
            match,
            odds,
            player_ratings=bundle["ratings"],
            esports_styles=bundle["styles"],
            patch_meta=bundle["patch_meta_by_match"].get(match.match_id),
        )
        fair = no_vig_probabilities_by_market(bundle["odds"].values())[odds.market_id]

        prediction = ModelDebatePredictor().predict_vector(vector, market_probability=fair)

        self.assertEqual(prediction.model_name, "debate-ensemble-v0.1")
        self.assertIn("market-no-vig", prediction.model_probabilities)
        self.assertGreaterEqual(prediction.disagreement, 0.0)
        self.assertIsNotNone(prediction.consensus_probability)
        self.assertIsNotNone(prediction.market_probability)
        self.assertLessEqual(prediction.confidence_low, prediction.probability)
        self.assertGreaterEqual(prediction.confidence_high, prediction.probability)

    def test_strategy_carries_debate_metadata_into_recommendation(self):
        prediction = Prediction(
            match_id="m1",
            model_name="debate",
            target_side="home",
            probability=0.62,
            confidence_low=0.52,
            confidence_high=0.70,
            factors={},
            disagreement=0.12,
            consensus_probability=0.66,
            veto_reasons=["high_model_disagreement"],
        )
        odds = MarketOdds("m1", "bet365", "moneyline", "home", 2.2, datetime.now(timezone.utc))

        recommendation = BankrollStrategy(RiskConfig(kelly_fraction=0.5, max_single_bet_fraction=0.2)).recommend(
            prediction,
            odds,
            bankroll=1000,
            fair_implied_probability=0.48,
        )

        self.assertEqual(recommendation.model_disagreement, 0.12)
        self.assertEqual(recommendation.consensus_probability, 0.66)
        self.assertIn("high_model_disagreement", recommendation.reasons)
        self.assertLess(recommendation.applied_fraction, recommendation.raw_kelly_fraction * 0.5)

    def test_portfolio_keeps_best_same_match_candidate_and_caps_budget(self):
        planner = PortfolioPlanner(PortfolioConfig(max_portfolio_fraction=0.03, max_match_fraction=0.02))
        recommendations = [
            self._recommendation("m1", "home", stake=20.0, ev=0.12, edge=0.05),
            self._recommendation("m1", "away", stake=20.0, ev=0.04, edge=0.02),
            self._recommendation("m2", "home", stake=25.0, ev=0.09, edge=0.04),
        ]

        planned = planner.plan(recommendations, bankroll=1000)

        self.assertEqual(planned[1].status, "skipped")
        self.assertIn("portfolio_same_match_lower_rank", planned[1].reasons)
        self.assertLessEqual(sum(item.stake for item in planned), 30.0)
        self.assertLessEqual(planned[0].stake, 20.0)

    def test_portfolio_scales_stake_when_models_disagree(self):
        planner = PortfolioPlanner(PortfolioConfig(max_portfolio_fraction=0.10, max_match_fraction=0.10))
        recommendation = self._recommendation("m1", "home", stake=50.0, ev=0.12, edge=0.05, disagreement=0.2)

        planned = planner.plan([recommendation], bankroll=1000)[0]

        self.assertLess(planned.stake, recommendation.stake)
        self.assertIn("portfolio_disagreement_scaled_stake", planned.reasons)

    @staticmethod
    def _recommendation(
        match_id: str,
        side: str,
        stake: float,
        ev: float,
        edge: float,
        disagreement: float = 0.0,
    ) -> BetRecommendation:
        return BetRecommendation(
            match_id=match_id,
            market_id=f"{match_id}-{side}",
            bookmaker="bet365",
            market_type="moneyline",
            side=side,  # type: ignore[arg-type]
            decimal_odds=2.0,
            model_probability=0.55,
            implied_probability=0.5,
            edge=edge,
            expected_value=ev,
            raw_kelly_fraction=0.1,
            applied_fraction=stake / 1000,
            stake=stake,
            risk_label="standard",
            status="recommended",
            model_disagreement=disagreement,
        )


if __name__ == "__main__":
    unittest.main()
