"""Pinned model snapshots and the dated pricing table."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

_PRICING_TABLE = Path(__file__).with_name("models.json")

SINGLE_CALL_BASELINE = "single-call baseline"
CHEAP_VISION = "cheap vision"
STRONG_VISION = "strong vision"
TEXT_LAYER = "text layer + text model"

BASELINE_MODEL_ENV = "NFE_BASELINE_MODEL"
CHEAP_VISION_MODEL_ENV = "NFE_CHEAP_VISION_MODEL"
STRONG_VISION_MODEL_ENV = "NFE_STRONG_VISION_MODEL"
TEXT_LAYER_MODEL_ENV = "NFE_TEXT_LAYER_MODEL"

_MODEL_ENV = {
    SINGLE_CALL_BASELINE: BASELINE_MODEL_ENV,
    CHEAP_VISION: CHEAP_VISION_MODEL_ENV,
    STRONG_VISION: STRONG_VISION_MODEL_ENV,
    TEXT_LAYER: TEXT_LAYER_MODEL_ENV,
}


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
    """Load one approach. Prices are USD per million tokens.

    The snapshot comes from ``models.json``. ``NFE_BASELINE_MODEL``,
    ``NFE_CHEAP_VISION_MODEL``, ``NFE_STRONG_VISION_MODEL``, and
    ``NFE_TEXT_LAYER_MODEL`` override it when set. The override must be a
    snapshot already priced in the table.
    """
    payload = _pricing_table()
    pinned = _approach_row(payload, approach)
    model = _selected_model(approach, str(pinned["model"]))
    priced = _model_row(payload, model)
    if priced is None:
        known = ", ".join(_known_models(payload))
        env_name = _MODEL_ENV[approach]
        raise ValueError(
            f"{env_name}={model} is not in the pricing table. Known snapshots: {known}."
        )
    return ApproachConfig(
        approach=str(pinned["approach"]),
        model=model,
        pricing_as_of=date.fromisoformat(str(payload["pricing_as_of"])),
        input_usd_per_million=Decimal(str(priced["input_usd_per_million"])),
        cached_input_usd_per_million=Decimal(
            str(priced["cached_input_usd_per_million"])
        ),
        output_usd_per_million=Decimal(str(priced["output_usd_per_million"])),
    )


def load_baseline_config() -> ApproachConfig:
    """The cheap-vision single-call baseline."""
    return load_approach_config(SINGLE_CALL_BASELINE)


def _pricing_table() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(_PRICING_TABLE.read_text(encoding="utf-8")))


def _approach_row(payload: dict[str, Any], approach: str) -> dict[str, Any]:
    for row in payload["approaches"]:
        if row["approach"] == approach:
            return cast(dict[str, Any], row)
    raise KeyError(approach)


def _model_row(payload: dict[str, Any], model: str) -> dict[str, Any] | None:
    for row in payload["approaches"]:
        if row["model"] == model:
            return cast(dict[str, Any], row)
    return None


def _known_models(payload: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted({str(row["model"]) for row in payload["approaches"]}))


def _selected_model(approach: str, pinned: str) -> str:
    override = os.environ.get(_MODEL_ENV[approach], "").strip()
    return override or pinned
