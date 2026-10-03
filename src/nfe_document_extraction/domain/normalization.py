"""Normalize DANFE values into the extraction schema."""

from __future__ import annotations

from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation


def normalize_digits(value: object) -> str:
    if isinstance(value, str):
        return "".join(character for character in value if character.isdigit())
    raise ValueError("expected a string of digits")


def normalize_numeric_identifier(value: object) -> str:
    if isinstance(value, str):
        return normalize_digits(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise ValueError("expected a numeric identifier")


def normalize_state(value: object) -> str:
    if isinstance(value, str):
        return value.strip().upper()
    raise ValueError("state must be a string")


def normalize_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    raise ValueError("expected text")


def normalize_issue_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return _parse_date_text(value)
    raise ValueError("issue date must be a date")


def normalize_money(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, Decimal):
        return _reais_to_cents(value)
    if isinstance(value, str):
        return _reais_to_cents(_parse_decimal_text(value))
    raise ValueError("money must be integer cents or a reais amount")


def normalize_decimal(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Decimal(value)
    if isinstance(value, str):
        return _parse_decimal_text(value)
    raise ValueError("expected an exact decimal")


def _parse_date_text(value: str) -> date:
    text = value.strip()
    if "/" in text:
        day_text, month_text, year_text = text.split("/")
        return date(int(year_text), int(month_text), int(day_text))
    if "T" in text or " " in text:
        return datetime.fromisoformat(text).date()
    return date.fromisoformat(text)


def _parse_decimal_text(value: str) -> Decimal:
    cleaned = value.strip().replace("R$", "").replace("\u00a0", "").replace(" ", "")
    if "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    try:
        return Decimal(cleaned)
    except InvalidOperation as exc:
        raise ValueError("expected a decimal amount") from exc


def _reais_to_cents(amount: Decimal) -> int:
    cents = (amount * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return int(cents)
