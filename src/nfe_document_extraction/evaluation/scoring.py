"""Pure scoring of an Extraction against ground truth."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from nfe_document_extraction.domain import Extraction, LineItem

# A description is correct at or above this normalized Levenshtein ratio.
DESCRIPTION_CORRECT_THRESHOLD = Decimal("0.95")

_HEADER_FIELDS = (
    "access_key",
    "state",
    "model",
    "series",
    "number",
    "issue_date",
    "issuer_cnpj",
    "recipient_cnpj_cpf",
    "total_products",
    "freight",
    "insurance",
    "discount",
    "ipi_total",
    "other_expenses",
    "icms_total",
    "icms_st_total",
    "invoice_total",
)
_UNSUPPORTED_FIELDS = (
    "import_tax",
    "returned_ipi",
    "icms_relief",
    "fcp_st",
)


@dataclass(frozen=True)
class DocumentScore:
    """How one Extraction compares with the printed ground truth."""

    field_correct: int
    field_opportunities: int
    missing_items: int
    spurious_items: int
    description_similarities: tuple[Decimal, ...]
    perfect: bool

    @property
    def field_accuracy(self) -> Decimal:
        if self.field_opportunities == 0:
            return Decimal(0)
        return Decimal(self.field_correct) / Decimal(self.field_opportunities)


def score_document(extracted: Extraction, ground_truth: Extraction) -> DocumentScore:
    """Score one reading. Missing and spurious items each count as one error."""
    pairs, missing_items, spurious_items = _align(
        extracted.line_items,
        ground_truth.line_items,
    )
    correct = 0
    opportunities = 0
    other_field_mismatch = False
    similarities: list[Decimal] = []

    for name in _HEADER_FIELDS:
        opportunities += 1
        if _equal(
            name, getattr(extracted.header, name), getattr(ground_truth.header, name)
        ):
            correct += 1
        else:
            other_field_mismatch = True
    for name in _UNSUPPORTED_FIELDS:
        opportunities += 1
        extracted_flag = getattr(extracted.header.unsupported_components, name)
        truth_flag = getattr(ground_truth.header.unsupported_components, name)
        if extracted_flag is truth_flag:
            correct += 1
        else:
            other_field_mismatch = True

    for extracted_item, truth_item in pairs:
        for name in LineItem.model_fields:
            opportunities += 1
            if name == "description":
                similarity = description_similarity(
                    extracted_item.description,
                    truth_item.description,
                )
                similarities.append(similarity)
                if _fold(extracted_item.description) == _fold(truth_item.description):
                    correct += 1
                continue
            if _equal(name, getattr(extracted_item, name), getattr(truth_item, name)):
                correct += 1
            else:
                other_field_mismatch = True

    opportunities += missing_items + spurious_items
    descriptions_correct = all(
        similarity >= DESCRIPTION_CORRECT_THRESHOLD for similarity in similarities
    )
    return DocumentScore(
        field_correct=correct,
        field_opportunities=opportunities,
        missing_items=missing_items,
        spurious_items=spurious_items,
        description_similarities=tuple(similarities),
        perfect=(
            missing_items == 0
            and spurious_items == 0
            and not other_field_mismatch
            and descriptions_correct
        ),
    )


def description_similarity(left: str, right: str) -> Decimal:
    """Normalized Levenshtein ratio after casefold, accent removal, and whitespace collapse."""
    folded_left = _fold(left)
    folded_right = _fold(right)
    if not folded_left and not folded_right:
        return Decimal(1)
    distance = _levenshtein(folded_left, folded_right)
    longest = max(len(folded_left), len(folded_right))
    return Decimal(longest - distance) / Decimal(longest)


def _align(
    extracted: tuple[LineItem, ...],
    ground_truth: tuple[LineItem, ...],
) -> tuple[tuple[tuple[LineItem, LineItem], ...], int, int]:
    """Order-preserving alignment on product code and description.

    Two items pair when their product codes match or their descriptions are
    correct at the documented threshold. On a tie, the earlier ground-truth
    item wins.
    """
    row_count = len(extracted) + 1
    column_count = len(ground_truth) + 1
    unreachable = (-1, -1, -1, -(10**9))
    scores = [[unreachable] * column_count for _ in range(row_count)]
    choices: list[list[str | None]] = [[None] * column_count for _ in range(row_count)]
    scores[0][0] = (0, 0, 0, 0)
    for extracted_index in range(row_count):
        for truth_index in range(column_count):
            if extracted_index == 0 and truth_index == 0:
                continue
            candidates: list[tuple[tuple[int, int, int, int], str]] = []
            if extracted_index > 0 and truth_index > 0:
                key = _pair_key(
                    extracted[extracted_index - 1],
                    ground_truth[truth_index - 1],
                )
                if key is not None:
                    previous = scores[extracted_index - 1][truth_index - 1]
                    candidates.append(
                        (
                            (
                                previous[0] + 1,
                                previous[1] + key[0],
                                previous[2] + key[1],
                                previous[3] - (truth_index - 1),
                            ),
                            "align",
                        )
                    )
            if extracted_index > 0:
                candidates.append(
                    (scores[extracted_index - 1][truth_index], "spurious")
                )
            if truth_index > 0:
                candidates.append((scores[extracted_index][truth_index - 1], "missing"))
            best_score, best_choice = max(
                candidates,
                key=lambda candidate: (candidate[0], _CHOICE_PRIORITY[candidate[1]]),
            )
            scores[extracted_index][truth_index] = best_score
            choices[extracted_index][truth_index] = best_choice

    steps: list[tuple[str, int, int]] = []
    extracted_index = len(extracted)
    truth_index = len(ground_truth)
    while extracted_index > 0 or truth_index > 0:
        choice = choices[extracted_index][truth_index]
        if choice == "align":
            steps.append(("align", extracted_index - 1, truth_index - 1))
            extracted_index -= 1
            truth_index -= 1
        elif choice == "spurious":
            steps.append(("spurious", extracted_index - 1, -1))
            extracted_index -= 1
        else:
            steps.append(("missing", -1, truth_index - 1))
            truth_index -= 1
    steps.reverse()
    pairs = tuple(
        (extracted[left], ground_truth[right])
        for kind, left, right in steps
        if kind == "align"
    )
    missing_items = sum(kind == "missing" for kind, _, _ in steps)
    spurious_items = sum(kind == "spurious" for kind, _, _ in steps)
    return pairs, missing_items, spurious_items


_CHOICE_PRIORITY = {"align": 2, "missing": 1, "spurious": 0}


def _pair_key(extracted: LineItem, ground_truth: LineItem) -> tuple[int, int] | None:
    code_match = _fold(extracted.product_code) == _fold(ground_truth.product_code)
    similarity = description_similarity(extracted.description, ground_truth.description)
    if not code_match and similarity < DESCRIPTION_CORRECT_THRESHOLD:
        return None
    similarity_milli = int(
        (similarity * 1000).to_integral_value(rounding=ROUND_HALF_UP)
    )
    return int(code_match), similarity_milli


def _equal(name: str, extracted: object, ground_truth: object) -> bool:
    if name in {"product_code", "description"} and isinstance(extracted, str):
        return _fold(extracted) == _fold(str(ground_truth))
    return extracted == ground_truth


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold())
    without_accents = "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )
    return " ".join(without_accents.split())


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index, left_character in enumerate(left, start=1):
        current = [index]
        for right_index, right_character in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (
                left_character != right_character
            )
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]
