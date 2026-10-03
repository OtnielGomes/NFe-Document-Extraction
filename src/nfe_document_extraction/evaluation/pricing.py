"""Turn recorded token counts into USD using a dated pricing table."""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol


class TokenPrice(Protocol):
    @property
    def input_usd_per_million(self) -> Decimal: ...

    @property
    def cached_input_usd_per_million(self) -> Decimal: ...

    @property
    def output_usd_per_million(self) -> Decimal: ...


def token_cost_usd(
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    price: TokenPrice,
) -> Decimal:
    """Price one model call. Cached input is billed at the cached rate."""
    cached = min(max(cached_input_tokens, 0), max(input_tokens, 0))
    uncached = max(input_tokens, 0) - cached
    million = Decimal(1_000_000)
    return (
        Decimal(uncached) * price.input_usd_per_million
        + Decimal(cached) * price.cached_input_usd_per_million
        + Decimal(max(output_tokens, 0)) * price.output_usd_per_million
    ) / million
