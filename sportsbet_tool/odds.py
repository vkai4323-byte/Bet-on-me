from __future__ import annotations

from collections.abc import Iterable


def validate_decimal_odds(decimal_odds: float) -> None:
    if decimal_odds <= 1.0:
        raise ValueError("decimal_odds must be greater than 1.0")


def implied_probability(decimal_odds: float) -> float:
    validate_decimal_odds(decimal_odds)
    return 1.0 / decimal_odds


def no_vig_probabilities(decimal_odds_values: Iterable[float]) -> list[float]:
    raw = [implied_probability(value) for value in decimal_odds_values]
    total = sum(raw)
    if total <= 0:
        raise ValueError("odds set has no probability mass")
    return [value / total for value in raw]


def expected_value(model_probability: float, decimal_odds: float) -> float:
    validate_decimal_odds(decimal_odds)
    if not 0.0 <= model_probability <= 1.0:
        raise ValueError("model_probability must be between 0 and 1")
    return model_probability * (decimal_odds - 1.0) - (1.0 - model_probability)


def decimal_kelly_fraction(model_probability: float, decimal_odds: float) -> float:
    validate_decimal_odds(decimal_odds)
    if not 0.0 <= model_probability <= 1.0:
        raise ValueError("model_probability must be between 0 and 1")
    b = decimal_odds - 1.0
    q = 1.0 - model_probability
    fraction = (b * model_probability - q) / b
    return max(0.0, fraction)
