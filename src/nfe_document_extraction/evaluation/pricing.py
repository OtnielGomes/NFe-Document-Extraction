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

    @property
    def cache_write_usd_per_million(self) -> Decimal: ...


def token_cost_usd(
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    price: TokenPrice,
    cache_write_tokens: int = 0,
) -> Decimal:
    """Price one model call.

    Cached input and cache writes are parts of ``input_tokens`` and are billed
    at their own rates instead of the uncached input rate. A row with no
    cache-write price bills those tokens at zero.
    """
    total = max(input_tokens, 0)
    cached = min(max(cached_input_tokens, 0), total)
    writes = min(max(cache_write_tokens, 0), total - cached)
    uncached = total - cached - writes
    million = Decimal(1_000_000)
    return (
        Decimal(uncached) * price.input_usd_per_million
        + Decimal(cached) * price.cached_input_usd_per_million
        + Decimal(writes) * price.cache_write_usd_per_million
        + Decimal(max(output_tokens, 0)) * price.output_usd_per_million
    ) / million
