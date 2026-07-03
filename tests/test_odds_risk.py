import unittest
from datetime import datetime, timezone

from sportsbet_tool.models import MarketOdds, Prediction
from sportsbet_tool.odds import decimal_kelly_fraction, expected_value, implied_probability, no_vig_probabilities_by_market
from sportsbet_tool.risk import BankrollStrategy, RiskConfig


class OddsRiskTests(unittest.TestCase):
    def test_decimal_odds_math(self):
        self.assertAlmostEqual(implied_probability(2.0), 0.5)
        self.assertAlmostEqual(expected_value(0.55, 2.0), 0.10)
        self.assertGreater(decimal_kelly_fraction(0.55, 2.0), 0.0)

    def test_strategy_caps_single_bet_and_uses_bet365(self):
        prediction = Prediction(
            match_id="m1",
            model_name="test",
            target_side="home",
            probability=0.62,
            confidence_low=0.5,
            confidence_high=0.7,
            factors={},
        )
        odds = MarketOdds(
            match_id="m1",
            bookmaker="bet365",
            market_type="moneyline",
            side="home",
            decimal_odds=2.1,
            observed_at=datetime.now(timezone.utc),
        )
        strategy = BankrollStrategy(RiskConfig(kelly_fraction=0.5, max_single_bet_fraction=0.02))
        recommendation = strategy.recommend(prediction, odds, bankroll=1000)
        self.assertEqual(recommendation.bookmaker, "bet365")
        self.assertEqual(recommendation.status, "recommended")
        self.assertLessEqual(recommendation.stake, 20.0)
        self.assertTrue(recommendation.requires_manual_confirmation)

    def test_strategy_uses_no_vig_implied_probability(self):
        observed_at = datetime.now(timezone.utc)
        odds = MarketOdds("m1", "bet365", "moneyline", "home", 2.0, observed_at, market_id="home")
        away = MarketOdds("m1", "bet365", "moneyline", "away", 1.8, observed_at, market_id="away")
        fair = no_vig_probabilities_by_market([odds, away])
        prediction = Prediction("m1", "test", "home", 0.55, 0.5, 0.7, {})
        recommendation = BankrollStrategy(RiskConfig()).recommend(
            prediction,
            odds,
            bankroll=1000,
            fair_implied_probability=fair["home"],
        )
        self.assertAlmostEqual(recommendation.book_implied_probability, 0.5)
        self.assertAlmostEqual(recommendation.implied_probability, fair["home"])
        self.assertIn("no_vig_implied_probability", recommendation.reasons)

    def test_consecutive_losses_pause(self):
        prediction = Prediction("m1", "test", "home", 0.7, 0.6, 0.8, {})
        odds = MarketOdds("m1", "bet365", "moneyline", "home", 2.0, datetime.now(timezone.utc))
        strategy = BankrollStrategy(RiskConfig(pause_after_consecutive_losses=3))
        recommendation = strategy.recommend(prediction, odds, bankroll=1000, consecutive_losses=3)
        self.assertEqual(recommendation.status, "paused")
        self.assertEqual(recommendation.stake, 0.0)

    def test_daily_cap_reason_is_explicit(self):
        prediction = Prediction("m1", "test", "home", 0.7, 0.6, 0.8, {})
        odds = MarketOdds("m1", "bet365", "moneyline", "home", 2.0, datetime.now(timezone.utc))
        strategy = BankrollStrategy(RiskConfig(max_daily_risk_fraction=0.05))
        recommendation = strategy.recommend(prediction, odds, bankroll=1000, daily_risk_used=50)
        self.assertEqual(recommendation.status, "skipped")
        self.assertIn("daily_risk_cap_reached", recommendation.reasons)

    def test_low_quality_skips_recommendation(self):
        prediction = Prediction("m1", "test", "home", 0.7, 0.6, 0.8, {})
        odds = MarketOdds("m1", "bet365", "moneyline", "home", 2.0, datetime.now(timezone.utc))
        recommendation = BankrollStrategy(RiskConfig(min_quality_score=0.55)).recommend(
            prediction,
            odds,
            bankroll=1000,
            quality_score=0.4,
            quality_reasons=["missing_core_factors:elo_delta"],
        )
        self.assertEqual(recommendation.status, "skipped")
        self.assertEqual(recommendation.quality_score, 0.4)
        self.assertIn("quality_below_threshold", recommendation.reasons)


if __name__ == "__main__":
    unittest.main()
