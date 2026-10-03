"""CNPJ, CPF, and access key checks."""

from __future__ import annotations

from dataclasses import dataclass

# IBGE codes used in the access key. 99 (exterior) is compared as digits.
_UF_CODE_BY_ABBREVIATION: dict[str, str] = {
    "AC": "12",
    "AL": "27",
    "AM": "13",
    "AP": "16",
    "BA": "29",
    "CE": "23",
    "DF": "53",
    "ES": "32",
    "GO": "52",
    "MA": "21",
    "MG": "31",
    "MS": "50",
    "MT": "51",
    "PA": "15",
    "PB": "25",
    "PE": "26",
    "PI": "22",
    "PR": "41",
    "RJ": "33",
    "RN": "24",
    "RO": "11",
    "RR": "14",
    "RS": "43",
    "SC": "42",
    "SE": "28",
    "SP": "35",
    "TO": "17",
}

_CNPJ_FIRST_WEIGHTS = (5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
_CNPJ_SECOND_WEIGHTS = (6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
_CPF_FIRST_WEIGHTS = (10, 9, 8, 7, 6, 5, 4, 3, 2)
_CPF_SECOND_WEIGHTS = (11, 10, 9, 8, 7, 6, 5, 4, 3, 2)
_ACCESS_KEY_WEIGHTS = (2, 3, 4, 5, 6, 7, 8, 9)


@dataclass(frozen=True)
class AccessKeyParts:
    state: str
    year_month: str
    issuer_cnpj: str
    model: str
    series: str
    number: str
    check_digit: str


def parse_access_key(access_key: str) -> AccessKeyParts | None:
    if len(access_key) != 44 or not access_key.isdigit():
        return None
    return AccessKeyParts(
        state=access_key[0:2],
        year_month=access_key[2:6],
        issuer_cnpj=access_key[6:20],
        model=access_key[20:22],
        series=access_key[22:25],
        number=access_key[25:34],
        check_digit=access_key[43],
    )


def access_key_check_digit(access_key: str) -> str | None:
    if parse_access_key(access_key) is None:
        return None
    total = sum(
        int(digit) * _ACCESS_KEY_WEIGHTS[index % len(_ACCESS_KEY_WEIGHTS)]
        for index, digit in enumerate(reversed(access_key[:43]))
    )
    return _digit_from_total(total)


def cnpj_check_digits(cnpj: str) -> str:
    first = _weighted_digit(cnpj[:12], _CNPJ_FIRST_WEIGHTS)
    second = _weighted_digit(cnpj[:12] + first, _CNPJ_SECOND_WEIGHTS)
    return first + second


def cpf_check_digits(cpf: str) -> str:
    first = _weighted_digit(cpf[:9], _CPF_FIRST_WEIGHTS)
    second = _weighted_digit(cpf[:9] + first, _CPF_SECOND_WEIGHTS)
    return first + second


def states_match(header_state: str, key_state: str) -> bool:
    mapped = _UF_CODE_BY_ABBREVIATION.get(header_state)
    if mapped is not None:
        return mapped == key_state
    return header_state == key_state


def same_numeric_identifier(left: str, right: str) -> bool:
    return bool(left.isdigit() and right.isdigit() and int(left) == int(right))


def _weighted_digit(numbers: str, weights: tuple[int, ...]) -> str:
    total = sum(
        int(digit) * weight for digit, weight in zip(numbers, weights, strict=True)
    )
    return _digit_from_total(total)


def _digit_from_total(total: int) -> str:
    remainder = total % 11
    if remainder < 2:
        return "0"
    return str(11 - remainder)
