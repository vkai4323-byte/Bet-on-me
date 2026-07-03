from __future__ import annotations

import unittest

from sportsbet_tool.research import audit_decision_card, build_research_brief, plan_from_research_card
from sportsbet_tool.tool_api import plan_research_card, research_brief


class ResearchWorkflowTests(unittest.TestCase):
    def test_brief_guides_ai_thinking_not_live_fetching(self):
        brief = build_research_brief("10 USDT adventurous World Cup card", bankroll=10, risk_mode="adventurous")

        self.assertEqual(brief.bankroll, 10)
        self.assertEqual(brief.risk_mode, "adventurous")
        self.assertIn("does not fetch live data", brief.ai_role)
        steps = {item.name for item in brief.thinking_steps}
        self.assertIn("evidence_collection", steps)
        self.assertIn("prediction_debate", steps)
        self.assertIn("market_translation", steps)
        self.assertIn("analyst_summary", brief.decision_card_schema)

    def test_audit_requires_counterarguments_and_market_rationale(self):
        weak_card = {
            "bankroll": 10,
            "risk_mode": "adventurous",
            "matches": [{"match_id": "m1"}],
            "markets": [{"match_id": "m1", "market_id": "m1-home", "market_type": "moneyline"}],
        }

        audit = audit_decision_card(weak_card)

        self.assertEqual(audit["severity"], "weak")
        self.assertIn("missing_sources", audit["quality_reasons"])
        self.assertTrue(any(item.endswith("evidence_against") for item in audit["missing"]))

    def test_plan_from_decision_card_returns_trackable_recommendations(self):
        result = plan_from_research_card(_complete_card())

        self.assertEqual(result["bankroll"], 10)
        self.assertEqual(result["thinking_audit"]["severity"], "complete")
        self.assertTrue(result["manual_confirmation_required"])
        self.assertEqual(len(result["recommendations"]), 1)
        self.assertEqual(result["recommendations"][0]["market_id"], "m1-home-ml")
        self.assertGreaterEqual(result["total_stake"], 0)

    def test_tool_api_exposes_ai_planning_workflow(self):
        brief_payload = research_brief("safe look at tonight", bankroll=20, risk_mode="insurance")

        self.assertEqual(brief_payload["bankroll"], 20)
        self.assertIn("thinking_steps", brief_payload)

        result = plan_research_card(
            {
                "bankroll": 20,
                "risk_mode": "insurance",
                "matches": [],
                "markets": [],
                "sources": [],
                "assumptions": [],
            }
        )
        self.assertEqual(result["recommendations"], [])
        self.assertIn("thinking_audit", result)


def _complete_card() -> dict:
    return {
        "bankroll": 10,
        "risk_mode": "adventurous",
        "analyst_summary": {
            "thesis": "A has enough edge to back the win market.",
            "main_risks": ["Lineup uncertainty"],
            "why_not_no_bet": "The price remains above the conservative fair line.",
        },
        "matches": [
            {
                "match_id": "m1",
                "home": "A",
                "away": "B",
                "predicted_score": "2-1",
                "winner_lean": "A",
                "evidence_for": ["A has better recent form"],
                "evidence_against": ["B is dangerous in transition"],
                "failure_modes": ["A concedes first and cannot control tempo"],
                "confidence_note": "Moderate confidence only.",
            }
        ],
        "markets": [
            {
                "match_id": "m1",
                "market_id": "m1-home-ml",
                "market_type": "moneyline",
                "selection": "A win",
                "decimal_odds": 2.1,
                "fair_implied_probability": 0.45,
                "model_probability": 0.55,
                "quality_score": 0.8,
                "disagreement": 0.05,
                "bookmaker": "manual",
                "source_url": "https://example.test/odds",
                "why_this_market": "It best expresses the 2-1 winner lean without needing a margin.",
                "failure_mode": "A draws despite having more chances.",
                "rejected_alternatives": ["A -1.5 is too scoreline-sensitive"],
            }
        ],
        "sources": [{"title": "odds", "url": "https://example.test/odds", "notes": "manual test"}],
        "assumptions": ["manual card test"],
    }


if __name__ == "__main__":
    unittest.main()
