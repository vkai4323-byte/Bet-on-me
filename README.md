# Bet-on-me
买点玩玩咯

## SportsBet Tool

Local agent tool for sports and esports match prediction, backtesting, bankroll sizing, safe bet-slip preparation, and settlement tracking.

This project is built around three guardrails:

- It is a decision-support and research tool, not a guaranteed profit system.
- Real-money use must be legal in the user's jurisdiction and allowed by the sportsbook terms.
- Browser automation never submits the final bet. The tool can open pages, inspect visible markets, and prepare a slip, but the user must manually confirm.

## What v0.1 Supports

- Sports: football.
- Esports: LoL, Valorant, CS2.
- Factors:
  - Team and player Elo with rolling 7/14/30/90 day windows.
  - Opponent-strength correction.
  - Esports patch/meta, player style, champion/agent/map pool, head-to-head, BP fit, roster synergy.
  - Football referee, motivation, competition format, weather, venue, cards, suspensions, injuries, and schedule pressure.
- Modeling:
  - Deterministic heuristic probability model with calibration.
  - Stable interfaces for future scikit-learn or gradient boosting models.
- Bankroll:
  - Decimal odds.
  - 0.5 fractional Kelly default.
  - 2 percent max stake per bet.
  - 5 percent daily risk cap.
  - Consecutive-loss pause.
- Backtesting:
  - Time-ordered simulation.
  - ROI, hit rate, max drawdown, CLV, calibration buckets.
- Web Bridge:
  - Safe local adapter for opening a sportsbook page and preparing instructions.
  - Refuses final submit actions by design.
- Default public odds target:
  - `bet365` is used as the default bookmaker label and first sportsbook adapter target.

## Quick Start

Run the demo prediction:

```powershell
C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m sportsbet_tool.cli demo-predict --bankroll 1000
```

Run the sample backtest:

```powershell
C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m sportsbet_tool.cli backtest --bankroll 1000
```

Use it from another Agent/tool host:

```python
from sportsbet_tool.tool_api import predict_sample, backtest_sample

prediction_payload = predict_sample(bankroll=1000)
backtest_payload = backtest_sample(bankroll=1000)
```

Run tests:

```powershell
C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover -s tests
```

## Data Source Notes

The first version is free/public-source first. Production use should add source-specific adapters under `sportsbet_tool/data_ingestion.py` and persist immutable raw snapshots before feature generation.

Do not use live or in-play events in a pre-match model. The feature builder filters snapshots by `observed_at` and keeps pre-match football context separate from live match state.

## Safety Notes

`sportsbet_tool.execution_bridge.SafeSportsbookAdapter.submit_bet()` intentionally raises `SafetyError`. Keep it that way unless the project scope is explicitly changed after legal and platform review.
