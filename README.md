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
  - Debate ensemble that compares full-factor, strength-only, context-shock, and style/tactics model views.
  - Market no-vig probability is used as a baseline challenger, not blended into the internal model consensus.
  - Conservative probability adjustment when model views disagree.
  - Stable interfaces for future scikit-learn or gradient boosting models.
- Bankroll:
  - Decimal odds.
  - No-vig fair probability when both sides of a market are available.
  - 0.5 fractional Kelly default.
  - Portfolio-level planning across candidate bets.
  - Same-match filtering, exposure caps, and model-disagreement stake scaling.
  - 2 percent max stake per bet.
  - 5 percent daily risk cap.
  - Consecutive-loss pause.
- Data quality:
  - Core factor coverage checks by sport.
  - Low sample, weak patch data, and missing factor penalties.
  - Stake scaling or skipping when evidence quality is low.
- Backtesting:
  - Time-ordered simulation.
  - ROI, hit rate, max drawdown, CLV, calibration buckets.
- Web Bridge:
  - Safe local adapter for opening a sportsbook page and preparing instructions.
  - Refuses final submit actions by design.
- Default public odds target:
  - `bet365` is used as the default bookmaker label and first sportsbook adapter target.

## Quick Start

In Codex, use plain language first:

```text
帮我保守看一下有没有值得押的，1000本金。
```

```text
激进一点跑个回测，用 2k 资金。
```

```text
把全部候选都给我看，不只看推荐。
```

The `sportsbet-assistant` Codex skill maps wording such as `稳一点`, `激进`, `梭哈`, `回测`, `全部`, and `500本金` into tool settings automatically.

From this folder, the shortest manual prediction command is:

```powershell
.\bet.ps1
```

Natural-language CLI fallback:

```powershell
.\bet.ps1 ask "帮我稳一点，500本金，只看推荐"
```

Backtest shortcut:

```powershell
.\bet.ps1 ask "激进一点跑个回测，用2k资金"
```

Prediction output now includes:

- `consensus_probability`: the internal model consensus before conservative adjustment.
- `model_disagreement`: spread between independent model views.
- `model_probabilities`: per-model probabilities, including the market no-vig challenger when available.
- `veto_reasons`: reasons to cut exposure when the model debate is weak or conflicts with market context.
- `portfolio_*` reasons: stake changes made by the portfolio planner after single-bet Kelly sizing.

Use it from another Agent/tool host:

```python
from sportsbet_tool.tool_api import run_intent, predict_sample, backtest_sample

payload = run_intent("帮我保守看一下有没有值得押的，1000本金")
prediction_payload = predict_sample(bankroll=1000, active_only=True, risk_mode="steady")
backtest_payload = backtest_sample(bankroll=1000)
```

## Factor Weights and Risk Modes

The baseline model keeps factor weights in `sportsbet_tool/weighting.py`.

Weight groups:

- Strength: Elo, rolling Elo, opponent-strength adjustment, sample size.
- Esports: patch fit, meta pool overlap, player style, hero/agent/map pool, H2H, BP fit, synergy.
- Football: motivation, schedule pressure, injuries, suspensions, referee bias, weather, venue, cards, set pieces.

Weights are not meant to stay fixed forever. `FactorWeights.update_from_outcome()` supports light post-match adjustment: factors that pointed in the right direction are nudged up, factors that pointed the wrong way are nudged down, with learning-rate and multiplier bounds. For production, keep separate weights by sport, league, market type, and model version.

Risk modes:

- `insurance`: high edge and quality threshold, small stakes, pauses quickly after losses.
- `steady`: default balanced mode, 0.5 Kelly, 2 percent single-bet cap, 5 percent daily cap.
- `adventurous`: lower edge threshold and higher exposure for stronger opinions.
- `wild`: full Kelly with large caps; useful for simulation stress tests, not recommended as a default real-money mode.

Run tests:

```powershell
C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover -s tests
```

## Data Source Notes

The first version is free/public-source first. Production use should add source-specific adapters under `sportsbet_tool/data_ingestion.py` and persist immutable raw snapshots before feature generation.

Do not use live or in-play events in a pre-match model. The feature builder filters snapshots by `observed_at` and keeps pre-match football context separate from live match state.

Recommended collection priority:

- Odds: full market outcome sets, opening odds, current odds, closing odds, line movement, suspended markets.
- Esports: roster, patch, map veto/BP, player style, champion/agent/map pool, head-to-head, role swaps, LAN/online context.
- Football: lineups, injuries, suspensions, referee profile, motivation, weather, venue, rest days, travel, cards, xG-style team quality.
- Quality: source timestamp, sample size, source reliability, and whether the factor was known before the market snapshot.

## Safety Notes

`sportsbet_tool.execution_bridge.SafeSportsbookAdapter.submit_bet()` intentionally raises `SafetyError`. Keep it that way unless the project scope is explicitly changed after legal and platform review.
