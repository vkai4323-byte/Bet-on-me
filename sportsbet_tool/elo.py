from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import pow

from sportsbet_tool.models import PlayerRatingSnapshot, Sport


def expected_score(rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + pow(10.0, (opponent_rating - rating) / 400.0))


def update_rating(rating: float, opponent_rating: float, score: float, k_factor: float = 32.0) -> float:
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be between 0 and 1")
    return rating + k_factor * (score - expected_score(rating, opponent_rating))


@dataclass(slots=True)
class EloEvent:
    entity_id: str
    opponent_id: str
    sport: Sport
    played_at: datetime
    score: float
    opponent_rating_hint: float = 1500.0
    weight: float = 1.0
    role: str | None = None
    map_name: str | None = None
    champion_or_agent: str | None = None


class RollingEloCalculator:
    def __init__(self, base_rating: float = 1500.0, k_factor: float = 32.0) -> None:
        self.base_rating = base_rating
        self.k_factor = k_factor

    def build_snapshots(
        self,
        events: list[EloEvent],
        as_of: datetime,
        windows_days: tuple[int, ...] = (7, 14, 30, 90),
    ) -> dict[str, PlayerRatingSnapshot]:
        by_entity: dict[str, list[EloEvent]] = {}
        for event in events:
            if event.played_at <= as_of:
                by_entity.setdefault(event.entity_id, []).append(event)

        snapshots: dict[str, PlayerRatingSnapshot] = {}
        for entity_id, entity_events in by_entity.items():
            entity_events.sort(key=lambda item: item.played_at)
            ratings = {window: self._rating_for_window(entity_events, as_of, window) for window in windows_days}
            recent_events = [event for event in entity_events if event.played_at >= as_of - timedelta(days=max(windows_days))]
            avg_opponent = (
                sum(event.opponent_rating_hint for event in recent_events) / len(recent_events)
                if recent_events
                else self.base_rating
            )
            adjustment = (avg_opponent - self.base_rating) * 0.08
            latest = recent_events[-1] if recent_events else entity_events[-1]
            snapshots[entity_id] = PlayerRatingSnapshot(
                entity_id=entity_id,
                sport=latest.sport,
                observed_at=as_of,
                elo_7d=ratings.get(7, self.base_rating),
                elo_14d=ratings.get(14, self.base_rating),
                elo_30d=ratings.get(30, self.base_rating),
                elo_90d=ratings.get(90, self.base_rating),
                opponent_strength_adjustment=adjustment,
                role=latest.role,
                map_name=latest.map_name,
                champion_or_agent=latest.champion_or_agent,
                sample_size=len(recent_events),
            )
        return snapshots

    def _rating_for_window(self, events: list[EloEvent], as_of: datetime, window_days: int) -> float:
        cutoff = as_of - timedelta(days=window_days)
        rating = self.base_rating
        for event in events:
            if event.played_at < cutoff or event.played_at > as_of:
                continue
            weighted_k = self.k_factor * max(0.0, event.weight)
            rating = update_rating(rating, event.opponent_rating_hint, event.score, weighted_k)
        return rating
