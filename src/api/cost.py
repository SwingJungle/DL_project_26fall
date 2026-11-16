"""Cost accounting (P1.4).

The token-plan gateway is subscription-based and D-2 (quota/rate limits) is not
confirmed yet, so the price table in ``config/models.json`` is empty and costs
are recorded as 0.0. The dollar guard becomes active as soon as prices are
filled in; until then the request/token caps are the binding hard stop.
"""
from __future__ import annotations

from .config import models_config


def price_table() -> dict:
    return models_config().get("pricing", {}).get("prices", {}) or {}


def has_prices() -> bool:
    return bool(price_table())


def compute_cost_usd(model_key: str, usage: dict | None) -> float:
    usage = usage or {}
    price = price_table().get(model_key)
    if not price:
        return 0.0
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return round(
        prompt / 1_000_000 * float(price.get("input_per_mtok", 0.0))
        + completion / 1_000_000 * float(price.get("output_per_mtok", 0.0)),
        8,
    )
