from __future__ import annotations

from datetime import datetime

from sportsbet_tool.models import BetRecord


class SettlementEngine:
    def settle(self, records: list[BetRecord], results_by_match: dict[str, str], settled_at: datetime) -> list[BetRecord]:
        settled: list[BetRecord] = []
        for record in records:
            if record.status != "open":
                settled.append(record)
                continue
            actual = results_by_match.get(record.match_id)
            if actual is None:
                settled.append(record)
                continue
            won = actual == record.side
            record.status = "won" if won else "lost"
            record.profit = round(record.stake * (record.decimal_odds - 1.0), 2) if won else -record.stake
            record.settled_at = settled_at
            settled.append(record)
        return settled


def profit_summary(records: list[BetRecord]) -> dict[str, float]:
    settled = [record for record in records if record.status in {"won", "lost", "void"}]
    stake = sum(record.stake for record in settled if record.status in {"won", "lost"})
    profit = sum(record.profit for record in settled)
    wins = sum(1 for record in settled if record.status == "won")
    losses = sum(1 for record in settled if record.status == "lost")
    return {
        "settled_bets": float(len(settled)),
        "wins": float(wins),
        "losses": float(losses),
        "total_stake": round(stake, 2),
        "total_profit": round(profit, 2),
        "roi": profit / stake if stake else 0.0,
    }
