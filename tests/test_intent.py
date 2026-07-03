from __future__ import annotations

import unittest

from sportsbet_tool.intent import parse_intent
from sportsbet_tool.cli import _normalize_argv
from sportsbet_tool.tool_api import run_intent


class IntentTests(unittest.TestCase):
    def test_parse_conservative_prediction_with_bankroll(self):
        intent = parse_intent("帮我稳一点，只看值得下的，500本金")

        self.assertEqual(intent.command, "predict")
        self.assertEqual(intent.risk_mode, "insurance")
        self.assertEqual(intent.bankroll, 500.0)
        self.assertTrue(intent.active_only)

    def test_parse_aggressive_backtest(self):
        intent = parse_intent("用2k资金激进一点跑个回测")

        self.assertEqual(intent.command, "backtest")
        self.assertEqual(intent.risk_mode, "adventurous")
        self.assertEqual(intent.bankroll, 2000.0)

    def test_parse_show_all_wild(self):
        intent = parse_intent("梭哈模式，全部都给我看")

        self.assertEqual(intent.risk_mode, "wild")
        self.assertFalse(intent.active_only)

    def test_parse_usdt_adventurous_request(self):
        intent = parse_intent("10 USDT adventurous plan")

        self.assertEqual(intent.bankroll, 10.0)
        self.assertEqual(intent.risk_mode, "adventurous")

    def test_run_intent_returns_interpreted_settings(self):
        payload = run_intent("保守一点，500本金，只看推荐")

        self.assertEqual(payload["interpreted_intent"]["risk_mode"], "insurance")
        self.assertEqual(payload["interpreted_intent"]["bankroll"], 500.0)
        self.assertTrue(payload["interpreted_intent"]["active_only"])
        self.assertIn("recommendations", payload)

    def test_cli_treats_unknown_text_as_intent(self):
        self.assertEqual(_normalize_argv(["帮我稳一点", "500本金"]), ["ask", "帮我稳一点", "500本金"])
        self.assertEqual(_normalize_argv(["--bankroll", "500", "稳一点"]), ["--bankroll", "500", "ask", "稳一点"])
        self.assertEqual(_normalize_argv(["backtest"]), ["backtest"])


if __name__ == "__main__":
    unittest.main()
