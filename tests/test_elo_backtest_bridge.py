import unittest
from datetime import datetime, timedelta, timezone

from sportsbet_tool.backtesting import BacktestEngine, BacktestSample
from sportsbet_tool.cli import _load_example_bundle
from sportsbet_tool.elo import EloEvent, RollingEloCalculator
from sportsbet_tool.execution_bridge import SafeSportsbookAdapter, SafetyError
from sportsbet_tool.features import FeatureBuilder
from sportsbet_tool.tool_api import backtest_sample, predict_sample


class EloBacktestBridgeTests(unittest.TestCase):
    def test_rolling_elo_ignores_future_events(self):
        now = datetime(2026, 7, 1, tzinfo=timezone.utc)
        events = [
            EloEvent("A", "B", "lol", now - timedelta(days=2), 1.0, opponent_rating_hint=1500),
            EloEvent("A", "B", "lol", now + timedelta(days=1), 0.0, opponent_rating_hint=1500),
        ]
        snapshot = RollingEloCalculator().build_snapshots(events, now)["A"]
        self.assertEqual(snapshot.sample_size, 1)
        self.assertGreater(snapshot.elo_7d, 1500)

    def test_backtest_runs_on_sample_data(self):
        bundle = _load_example_bundle()
        builder = FeatureBuilder()
        samples = []
        for row in bundle["historical_results"]:
            match = bundle["matches"][row["match_id"]]
            odds = bundle["odds"][row["market_id"]]
            samples.append(
                BacktestSample(
                    start_time=match.start_time,
                    vector=builder.build(
                        match,
                        odds,
                        player_ratings=bundle["ratings"],
                        esports_styles=bundle["styles"],
                        patch_meta=bundle["patch_meta_by_match"].get(match.match_id),
                        football_context=bundle["football_context_by_match"].get(match.match_id),
                    ),
                    odds=odds,
                    actual_side=row["actual_side"],
                    closing_decimal_odds=row["closing_decimal_odds"],
                )
            )
        result = BacktestEngine().run(samples, starting_bankroll=1000)
        self.assertGreaterEqual(result.bets, 0)
        self.assertIn("0.5-0.6", result.calibration_buckets)

    def test_bridge_refuses_final_submit(self):
        adapter = SafeSportsbookAdapter(client=None)
        with self.assertRaises(SafetyError):
            adapter.submit_bet()

    def test_tool_api_returns_json_payloads(self):
        predictions = predict_sample(bankroll=1000)
        active_predictions = predict_sample(bankroll=1000, active_only=True)
        backtest = backtest_sample(bankroll=1000)
        self.assertEqual(predictions["bookmaker"], "bet365")
        self.assertTrue(predictions["manual_confirmation_required"])
        self.assertGreaterEqual(len(predictions["recommendations"]), 1)
        recommended = [item for item in predictions["recommendations"] if item["status"] == "recommended"]
        self.assertEqual(len(recommended), 4)
        self.assertEqual(len(active_predictions["recommendations"]), 4)
        self.assertTrue(all(item["status"] == "recommended" for item in active_predictions["recommendations"]))
        self.assertEqual(backtest["bookmaker"], "bet365")
        self.assertTrue(backtest["manual_confirmation_required"])


if __name__ == "__main__":
    unittest.main()
