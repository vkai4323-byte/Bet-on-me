from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import exp

from sportsbet_tool.models import (
    EsportsStyleSnapshot,
    FootballContextSnapshot,
    MarketOdds,
    Match,
    PatchMetaSnapshot,
    PlayerRatingSnapshot,
)


@dataclass(slots=True)
class FeatureVector:
    match_id: str
    sport: str
    target_side: str
    as_of: datetime
    features: dict[str, float]


def _is_visible(observed_at: datetime, as_of: datetime) -> bool:
    return observed_at <= as_of


def _safe(value: float | None, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _logistic_scale(value: float, scale: float = 1.0) -> float:
    return 2.0 / (1.0 + exp(-value / scale)) - 1.0


class FeatureBuilder:
    """Builds leakage-aware numeric features from sport-specific snapshots."""

    def build(
        self,
        match: Match,
        odds: MarketOdds,
        player_ratings: dict[str, PlayerRatingSnapshot] | None = None,
        esports_styles: dict[str, EsportsStyleSnapshot] | None = None,
        patch_meta: PatchMetaSnapshot | None = None,
        football_context: FootballContextSnapshot | None = None,
        as_of: datetime | None = None,
    ) -> FeatureVector:
        as_of = as_of or odds.observed_at
        features = self._base_features(match, odds)

        ratings = player_ratings or {}
        home_rating = ratings.get(match.home)
        away_rating = ratings.get(match.away)
        if home_rating and away_rating and _is_visible(home_rating.observed_at, as_of) and _is_visible(away_rating.observed_at, as_of):
            features.update(self._rating_features(home_rating, away_rating))

        if match.sport in {"lol", "valorant", "cs2"}:
            features.update(self._esports_features(match, esports_styles or {}, patch_meta, as_of))
        elif match.sport == "football" and football_context and _is_visible(football_context.observed_at, as_of):
            features.update(self._football_features(football_context, pre_match=not football_context.is_live))

        return FeatureVector(match.match_id, match.sport, odds.side, as_of, self._orient_features(features, odds.side))

    def _base_features(self, match: Match, odds: MarketOdds) -> dict[str, float]:
        return {
            "is_home_side": 1.0 if odds.side == "home" else 0.0,
            "is_away_side": 1.0 if odds.side == "away" else 0.0,
            "neutral_site": 1.0 if match.neutral_site else 0.0,
            "best_of": float(match.best_of),
            "market_line": _safe(odds.line),
        }

    def _rating_features(self, home_rating: PlayerRatingSnapshot, away_rating: PlayerRatingSnapshot) -> dict[str, float]:
        return {
            "elo_delta": home_rating.blended_elo - away_rating.blended_elo,
            "elo_7d_delta": home_rating.elo_7d - away_rating.elo_7d,
            "elo_30d_delta": home_rating.elo_30d - away_rating.elo_30d,
            "opponent_strength_delta": home_rating.opponent_strength_adjustment - away_rating.opponent_strength_adjustment,
            "rating_sample_delta": float(home_rating.sample_size - away_rating.sample_size),
            "rating_min_sample": float(min(home_rating.sample_size, away_rating.sample_size)),
        }

    def _esports_features(
        self,
        match: Match,
        styles: dict[str, EsportsStyleSnapshot],
        patch_meta: PatchMetaSnapshot | None,
        as_of: datetime,
    ) -> dict[str, float]:
        features: dict[str, float] = {}
        home = styles.get(match.home)
        away = styles.get(match.away)
        if home and away and _is_visible(home.observed_at, as_of) and _is_visible(away.observed_at, as_of):
            features.update(
                {
                    "style_aggression_delta": home.aggression - away.aggression,
                    "style_resource_delta": home.resource_share - away.resource_share,
                    "style_first_action_delta": home.first_action_rate - away.first_action_rate,
                    "style_clutch_delta": home.clutch_or_late_game - away.clutch_or_late_game,
                    "style_info_delta": home.vision_or_info - away.vision_or_info,
                    "style_teamfight_delta": home.teamfight_or_trade - away.teamfight_or_trade,
                    "style_consistency_delta": home.consistency - away.consistency,
                    "pool_depth_delta": home.hero_pool_depth - away.hero_pool_depth,
                    "meta_mastery_delta": home.meta_pool_mastery - away.meta_pool_mastery,
                    "map_pool_delta": home.map_pool_depth - away.map_pool_depth,
                    "h2h_delta": home.h2h_rating - away.h2h_rating,
                    "bp_fit_delta": home.bp_fit - away.bp_fit,
                    "synergy_delta": home.synergy - away.synergy,
                    "style_sample_delta": float(home.sample_size - away.sample_size),
                    "style_min_sample": float(min(home.sample_size, away.sample_size)),
                }
            )
            features["style_clash_home_advantage"] = self._style_clash(home, away)

        if patch_meta and _is_visible(patch_meta.observed_at, as_of):
            features.update(
                {
                    "patch_meta_shift": patch_meta.meta_shift_severity,
                    "patch_fit_delta": patch_meta.home_patch_fit - patch_meta.away_patch_fit,
                    "patch_pool_overlap_delta": patch_meta.strong_pool_overlap_home - patch_meta.strong_pool_overlap_away,
                    "patch_data_weight": patch_meta.data_weight,
                }
            )
        return features

    def _football_features(self, context: FootballContextSnapshot, pre_match: bool) -> dict[str, float]:
        features = {
            "referee_yellow_rate": context.referee_yellows_per_match,
            "referee_red_rate": context.referee_reds_per_match,
            "referee_penalty_rate": context.referee_penalties_per_match,
            "referee_home_bias": context.referee_home_bias,
            "motivation_delta": context.home_motivation - context.away_motivation,
            "format_pressure": context.format_pressure,
            "schedule_pressure_delta": context.away_schedule_pressure - context.home_schedule_pressure,
            "injury_impact_delta": context.away_injury_impact - context.home_injury_impact,
            "suspension_impact_delta": context.away_suspension_impact - context.home_suspension_impact,
            "weather_disruption": context.weather_disruption,
            "venue_home_familiarity": context.venue_home_familiarity,
            "card_tendency_delta": context.away_card_tendency - context.home_card_tendency,
            "set_piece_delta": context.home_set_piece_edge - context.away_set_piece_edge,
        }
        if not pre_match:
            features.update(
                {
                    "live_red_card_delta": float(context.live_away_red_cards - context.live_home_red_cards),
                    "live_yellow_card_delta": float(context.live_away_yellow_cards - context.live_home_yellow_cards),
                }
            )
        return features

    def _style_clash(self, home: EsportsStyleSnapshot, away: EsportsStyleSnapshot) -> float:
        tempo_edge = home.first_action_rate - away.first_action_rate
        information_edge = home.vision_or_info - away.vision_or_info
        late_game_edge = home.clutch_or_late_game - away.clutch_or_late_game
        return _logistic_scale(0.4 * tempo_edge + 0.35 * information_edge + 0.25 * late_game_edge, scale=1.0)

    def _orient_features(self, features: dict[str, float], side: str) -> dict[str, float]:
        if side == "away":
            oriented: dict[str, float] = {}
            for key, value in features.items():
                if key.endswith("_delta") or key in {
                    "elo_delta",
                    "referee_home_bias",
                    "venue_home_familiarity",
                    "style_clash_home_advantage",
                }:
                    oriented[key] = -value
                elif key in {"is_home_side", "is_away_side"}:
                    oriented[key] = 1.0 - value
                else:
                    oriented[key] = value
            return oriented
        return features
