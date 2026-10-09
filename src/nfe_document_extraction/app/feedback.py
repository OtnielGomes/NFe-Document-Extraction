"""Visitor comments, with invoice identifiers removed before they leave the process."""

from __future__ import annotations

_SEPARATORS = frozenset("./- ")
_MIN_DIGITS = 11
_MAX_LENGTH = 500
_REDACTED = "[redacted]"


def redact_feedback_comment(comment: str) -> str:
    """Replace 11-digit runs, ignoring ``.`` ``/`` ``-`` and spaces, then cut to 500.

    Shorter numbers stay. Separators count as part of a run only between its
    digits, so a period after the number is left in place.
    """
    pieces: list[str] = []
    index = 0
    length = len(comment)
    while index < length:
        character = comment[index]
        if _is_digit(character):
            last_digit = index
            digits = 0
            cursor = index
            while cursor < length and (
                _is_digit(comment[cursor]) or comment[cursor] in _SEPARATORS
            ):
                if _is_digit(comment[cursor]):
                    digits += 1
                    last_digit = cursor
                cursor += 1
            if digits >= _MIN_DIGITS:
                pieces.append(_REDACTED)
            else:
                pieces.append(comment[index : last_digit + 1])
            index = last_digit + 1
            continue
        pieces.append(character)
        index += 1
    return "".join(pieces)[:_MAX_LENGTH]


def _is_digit(character: str) -> bool:
    return character.isascii() and character.isdigit()
