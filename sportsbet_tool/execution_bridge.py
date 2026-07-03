from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from sportsbet_tool.models import BetRecommendation


class SafetyError(RuntimeError):
    pass


class WebBridgeError(RuntimeError):
    pass


@dataclass(slots=True)
class PreparedBetSlip:
    url: str
    recommendations: list[BetRecommendation]
    instructions: list[str]
    snapshot_text: str | None = None
    warnings: list[str] = field(default_factory=list)
    requires_manual_confirmation: bool = True


class WebBridgeClient:
    def __init__(self, endpoint: str = "http://127.0.0.1:10086/command", session: str = "sportsbet-tool") -> None:
        self.endpoint = endpoint
        self.session = session

    def command(self, action: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        body = json.dumps({"action": action, "args": args or {}, "session": self.session}).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise WebBridgeError(f"cannot reach Web Bridge endpoint: {exc}") from exc

    def navigate(self, url: str, group_title: str = "SportsBet Tool") -> dict[str, Any]:
        return self.command("navigate", {"url": url, "newTab": True, "group_title": group_title})

    def snapshot(self) -> dict[str, Any]:
        return self.command("snapshot", {})

    def fill(self, selector: str, value: str) -> dict[str, Any]:
        return self.command("fill", {"selector": selector, "value": value})


class SafeSportsbookAdapter:
    """Prepares bet-slip instructions but refuses final submit automation."""

    def __init__(self, client: WebBridgeClient | None = None) -> None:
        self.client = client or WebBridgeClient()

    def prepare_bet_slip(
        self,
        url: str,
        recommendations: list[BetRecommendation],
        stake_input_selector: str | None = None,
    ) -> PreparedBetSlip:
        valid = [item for item in recommendations if item.status == "recommended" and item.stake > 0]
        warnings: list[str] = []
        instructions = [
            "Confirm your jurisdiction and sportsbook terms before placing any real-money bet.",
            "Check that each market, side, odds, and stake exactly matches the recommendation.",
            "The tool stops before final confirmation. You must manually submit or cancel.",
        ]

        if not valid:
            warnings.append("No active recommendations to prepare.")

        self.client.navigate(url)
        snapshot_text: str | None = None
        try:
            snapshot = self.client.snapshot()
            snapshot_text = str(snapshot.get("tree", ""))
        except WebBridgeError as exc:
            warnings.append(str(exc))

        if stake_input_selector and len(valid) == 1:
            self.client.fill(stake_input_selector, f"{valid[0].stake:.2f}")
            instructions.append("Stake amount was prefilled. Verify it before any manual confirmation.")
        elif stake_input_selector and len(valid) > 1:
            warnings.append("Multiple recommendations were provided, so no single stake input was filled.")

        for item in valid:
            instructions.append(
                f"{item.bookmaker} {item.market_type} {item.side}: odds {item.decimal_odds:.3f}, stake {item.stake:.2f}, edge {item.edge:.3f}."
            )

        return PreparedBetSlip(
            url=url,
            recommendations=valid,
            instructions=instructions,
            snapshot_text=snapshot_text,
            warnings=warnings,
            requires_manual_confirmation=True,
        )

    def submit_bet(self, *_: Any, **__: Any) -> None:
        raise SafetyError("Final bet submission automation is disabled by design; manual confirmation is required.")


class Bet365SportsbookAdapter(SafeSportsbookAdapter):
    bookmaker = "bet365"

    def prepare_bet_slip(
        self,
        url: str,
        recommendations: list[BetRecommendation],
        stake_input_selector: str | None = None,
    ) -> PreparedBetSlip:
        filtered = [item for item in recommendations if item.bookmaker.lower() == self.bookmaker]
        slip = super().prepare_bet_slip(url, filtered, stake_input_selector)
        if len(filtered) != len(recommendations):
            slip.warnings.append("Non-bet365 recommendations were ignored by the bet365 adapter.")
        return slip
