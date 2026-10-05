"""In-memory upload and spend brakes for the public demo.

Counters reset at UTC midnight and whenever the process restarts. The
OpenAI project budget of US$ 5/month stays the last barrier: US$ 0.50
stops one day from taking the month, and 50 uploads stops a day on
which token usage was not recorded.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from nfe_document_extraction.app.errors import RequestRejected

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PAGES = 3
UPLOADS_PER_IP_PER_DAY = 5
GLOBAL_UPLOADS_PER_DAY = 50
DAILY_SPEND_CAP_USD = Decimal("0.50")


def _ip_limit_message(count: int) -> str:
    noun = "document" if count == 1 else "documents"
    return (
        f"This address has already uploaded {count} {noun} today. Try again tomorrow."
    )


@dataclass(frozen=True)
class CostLimits:
    """The brakes applied before a model call."""

    max_bytes: int = MAX_UPLOAD_BYTES
    max_pages: int = MAX_PAGES
    uploads_per_ip_per_day: int = UPLOADS_PER_IP_PER_DAY
    global_uploads_per_day: int = GLOBAL_UPLOADS_PER_DAY
    daily_spend_cap_usd: Decimal = DAILY_SPEND_CAP_USD


class UsageLedger:
    """Daily upload counts and estimated spend.

    An admission holds the lock until the caller records that upload's
    cost, so a second request cannot pass the spend brake first.
    """

    def __init__(self, limits: CostLimits, clock: Callable[[], date]) -> None:
        self.limits = limits
        self._clock = clock
        self._lock = threading.Lock()
        self._day: date | None = None
        self._per_ip: dict[str, int] = {}
        self._global = 0
        self._spend = Decimal(0)

    @contextmanager
    def admission(self, client_ip: str) -> Iterator[_Admission]:
        """Reserve one upload, or reject when a daily brake is already shut."""
        with self._lock:
            self._roll_forward()
            per_ip = self._per_ip.get(client_ip, 0)
            if per_ip >= self.limits.uploads_per_ip_per_day:
                raise RequestRejected(
                    _ip_limit_message(self.limits.uploads_per_ip_per_day),
                    429,
                )
            if self._global >= self.limits.global_uploads_per_day:
                raise RequestRejected(
                    "The demo has reached its daily upload limit. Try again tomorrow.",
                    429,
                )
            if self._spend >= self.limits.daily_spend_cap_usd:
                raise RequestRejected(
                    "The demo has reached its daily spend limit. Try again tomorrow.",
                    429,
                )
            self._per_ip[client_ip] = per_ip + 1
            self._global += 1
            yield _Admission(self)

    def _roll_forward(self) -> None:
        today = self._clock()
        if self._day == today:
            return
        self._day = today
        self._per_ip = {}
        self._global = 0
        self._spend = Decimal(0)


class _Admission:
    """The open reservation. Spend is added before the lock is released."""

    def __init__(self, ledger: UsageLedger) -> None:
        self._ledger = ledger

    def add_spend(self, amount: Decimal) -> None:
        self._ledger._spend += amount
