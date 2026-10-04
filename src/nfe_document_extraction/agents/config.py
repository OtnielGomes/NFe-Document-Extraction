"""Pinned model snapshots and the dated pricing table."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

_PRICING_TABLE = Path(__file__).with_name("models.json")

SINGLE_CALL_BASELINE = "single-call baseline"
CHEAP_VISION = "cheap vision"
STRONG_VISION = "strong vision"


@dataclass(frozen=True)
class ApproachConfig:
    """One extraction approach and the prices that turn its tokens into USD."""

    approach: str
    model: str
    pricing_as_of: date
    input_usd_per_million: Decimal
    cached_input_usd_per_million: Decimal
    output_usd_per_million: Decimal


def load_approach_config(approach: str) -> ApproachConfig:
    """Load one row of the repository pricing table. Prices are USD per million tokens."""
    payload = json.loads(_PRICING_TABLE.read_text(encoding="utf-8"))
    priced_on = date.fromisoformat(payload["pricing_as_of"])
    for row in payload["approaches"]:
        if row["approach"] == approach:
            return ApproachConfig(
                approach=row["approach"],
                model=row["model"],
                pricing_as_of=priced_on,
                input_usd_per_million=Decimal(row["input_usd_per_million"]),
                cached_input_usd_per_million=Decimal(
                    row["cached_input_usd_per_million"]
                ),
                output_usd_per_million=Decimal(row["output_usd_per_million"]),
            )
    raise KeyError(approach)


def load_baseline_config() -> ApproachConfig:
    """The cheap-vision single-call baseline."""
    return load_approach_config(SINGLE_CALL_BASELINE)
