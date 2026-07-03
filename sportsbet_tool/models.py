from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Sport = Literal["football", "lol", "valorant", "cs2"]
Side = Literal["home", "away", "draw", "over", "under", "handicap_home", "handicap_away"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    text = value.replace("Z", "+00:00")
    return datetime.fromisoformat(text)


def to_iso(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: to_iso(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [to_iso(inner) for inner in value]
    return value


def clean_dict(value: Any) -> dict[str, Any]:
    return to_iso(asdict(value))


@dataclass(slots=True)
class Weather:
    temperature_c: float | None = None
    rain_mm: float | None = None
    wind_kph: float | None = None
    humidity_pct: float | None = None


@dataclass(slots=True)
class Match:
    match_id: str
    sport: Sport
    league: str
    start_time: datetime
    home: str
    away: str
    best_of: int = 1
    stage: str = "regular"
    venue: str | None = None
    surface: str | None = None
    neutral_site: bool = False
    patch_version: str | None = None
    weather: Weather | None = None
    status: str = "scheduled"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MarketOdds:
    match_id: str
    bookmaker: str
    market_type: str
    side: Side
    decimal_odds: float
    observed_at: datetime
    line: float | None = None
    market_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlayerRatingSnapshot:
    entity_id: str
    sport: Sport
    observed_at: datetime
    elo_7d: float = 1500.0
    elo_14d: float = 1500.0
    elo_30d: float = 1500.0
    elo_90d: float = 1500.0
    opponent_strength_adjustment: float = 0.0
    role: str | None = None
    map_name: str | None = None
    champion_or_agent: str | None = None
    sample_size: int = 0

    @property
    def blended_elo(self) -> float:
        weighted = 0.35 * self.elo_30d + 0.25 * self.elo_14d + 0.25 * self.elo_90d + 0.15 * self.elo_7d
        return weighted + self.opponent_strength_adjustment


@dataclass(slots=True)
class EsportsStyleSnapshot:
    entity_id: str
    sport: Literal["lol", "valorant", "cs2"]
    observed_at: datetime
    aggression: float = 0.0
    resource_share: float = 0.0
    first_action_rate: float = 0.0
    clutch_or_late_game: float = 0.0
    vision_or_info: float = 0.0
    teamfight_or_trade: float = 0.0
    consistency: float = 0.0
    hero_pool_depth: float = 0.0
    meta_pool_mastery: float = 0.0
    map_pool_depth: float = 0.0
    h2h_rating: float = 0.0
    bp_fit: float = 0.0
    synergy: float = 0.0
    sample_size: int = 0


@dataclass(slots=True)
class PatchMetaSnapshot:
    sport: Literal["lol", "valorant", "cs2"]
    patch_version: str
    observed_at: datetime
    meta_shift_severity: float = 0.0
    home_patch_fit: float = 0.0
    away_patch_fit: float = 0.0
    strong_pool_overlap_home: float = 0.0
    strong_pool_overlap_away: float = 0.0
    data_weight: float = 1.0
    notes: str | None = None


@dataclass(slots=True)
class FootballContextSnapshot:
    match_id: str
    observed_at: datetime
    referee_yellows_per_match: float = 0.0
    referee_reds_per_match: float = 0.0
    referee_penalties_per_match: float = 0.0
    referee_home_bias: float = 0.0
    home_motivation: float = 0.0
    away_motivation: float = 0.0
    format_pressure: float = 0.0
    home_schedule_pressure: float = 0.0
    away_schedule_pressure: float = 0.0
    home_injury_impact: float = 0.0
    away_injury_impact: float = 0.0
    home_suspension_impact: float = 0.0
    away_suspension_impact: float = 0.0
    weather_disruption: float = 0.0
    venue_home_familiarity: float = 0.0
    home_card_tendency: float = 0.0
    away_card_tendency: float = 0.0
    home_set_piece_edge: float = 0.0
    away_set_piece_edge: float = 0.0
    live_home_red_cards: int = 0
    live_away_red_cards: int = 0
    live_home_yellow_cards: int = 0
    live_away_yellow_cards: int = 0
    is_live: bool = False


@dataclass(slots=True)
class Prediction:
    match_id: str
    model_name: str
    target_side: Side
    probability: float
    confidence_low: float
    confidence_high: float
    factors: dict[str, float]
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class BetRecommendation:
    match_id: str
    market_id: str | None
    bookmaker: str
    market_type: str
    side: Side
    decimal_odds: float
    model_probability: float
    implied_probability: float
    edge: float
    expected_value: float
    raw_kelly_fraction: float
    applied_fraction: float
    stake: float
    risk_label: str
    status: str
    reasons: list[str] = field(default_factory=list)
    requires_manual_confirmation: bool = True
    book_implied_probability: float | None = None
    quality_score: float = 1.0


@dataclass(slots=True)
class BetRecord:
    bet_id: str
    match_id: str
    market_id: str | None
    side: Side
    stake: float
    decimal_odds: float
    placed_at: datetime
    status: str = "open"
    profit: float = 0.0
    settled_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
