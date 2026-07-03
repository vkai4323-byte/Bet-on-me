from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, TypeVar

from sportsbet_tool.models import (
    EsportsStyleSnapshot,
    FootballContextSnapshot,
    MarketOdds,
    Match,
    PatchMetaSnapshot,
    PlayerRatingSnapshot,
    Weather,
    parse_dt,
)

T = TypeVar("T")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_records(path: str | Path, factory: Callable[[dict[str, Any]], T]) -> list[T]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("expected a JSON list")
    return [factory(item) for item in data]


def load_csv_records(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def match_from_dict(data: dict[str, Any]) -> Match:
    weather = data.get("weather")
    return Match(
        match_id=data["match_id"],
        sport=data["sport"],
        league=data["league"],
        start_time=parse_dt(data["start_time"]),  # type: ignore[arg-type]
        home=data["home"],
        away=data["away"],
        best_of=int(data.get("best_of", 1)),
        stage=data.get("stage", "regular"),
        venue=data.get("venue"),
        surface=data.get("surface"),
        neutral_site=bool(data.get("neutral_site", False)),
        patch_version=data.get("patch_version"),
        weather=Weather(**weather) if isinstance(weather, dict) else None,
        status=data.get("status", "scheduled"),
        metadata=data.get("metadata", {}),
    )


def odds_from_dict(data: dict[str, Any]) -> MarketOdds:
    return MarketOdds(
        match_id=data["match_id"],
        bookmaker=data["bookmaker"],
        market_type=data["market_type"],
        side=data["side"],
        decimal_odds=float(data["decimal_odds"]),
        observed_at=parse_dt(data["observed_at"]),  # type: ignore[arg-type]
        line=float(data["line"]) if data.get("line") is not None else None,
        market_id=data.get("market_id"),
        metadata=data.get("metadata", {}),
    )


def rating_from_dict(data: dict[str, Any]) -> PlayerRatingSnapshot:
    return PlayerRatingSnapshot(
        entity_id=data["entity_id"],
        sport=data["sport"],
        observed_at=parse_dt(data["observed_at"]),  # type: ignore[arg-type]
        elo_7d=float(data.get("elo_7d", 1500)),
        elo_14d=float(data.get("elo_14d", 1500)),
        elo_30d=float(data.get("elo_30d", 1500)),
        elo_90d=float(data.get("elo_90d", 1500)),
        opponent_strength_adjustment=float(data.get("opponent_strength_adjustment", 0)),
        role=data.get("role"),
        map_name=data.get("map_name"),
        champion_or_agent=data.get("champion_or_agent"),
        sample_size=int(data.get("sample_size", 0)),
    )


def style_from_dict(data: dict[str, Any]) -> EsportsStyleSnapshot:
    kwargs = {key: value for key, value in data.items() if key not in {"observed_at"}}
    kwargs["observed_at"] = parse_dt(data["observed_at"])
    return EsportsStyleSnapshot(**kwargs)


def patch_meta_from_dict(data: dict[str, Any]) -> PatchMetaSnapshot:
    kwargs = {key: value for key, value in data.items() if key not in {"observed_at"}}
    kwargs["observed_at"] = parse_dt(data["observed_at"])
    return PatchMetaSnapshot(**kwargs)


def football_context_from_dict(data: dict[str, Any]) -> FootballContextSnapshot:
    kwargs = {key: value for key, value in data.items() if key not in {"observed_at"}}
    kwargs["observed_at"] = parse_dt(data["observed_at"])
    return FootballContextSnapshot(**kwargs)
