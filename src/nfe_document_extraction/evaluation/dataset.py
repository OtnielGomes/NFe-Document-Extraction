"""Load the published synthetic set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nfe_document_extraction.domain import Extraction

DIGITAL_PDF = "digital PDF"
SIMULATED_PHOTO = "simulated photo"
PHONE_PHOTO = "phone photo"
PHOTO_RENDITIONS = frozenset({SIMULATED_PHOTO, PHONE_PHOTO})


@dataclass(frozen=True)
class DatasetExample:
    """One DANFE, its printed ground truth, and any planted error."""

    nfe_id: str
    origin: str
    pdf: bytes
    ground_truth: Extraction
    planted_rule: str | None
    rendition: str = DIGITAL_PDF


def load_synthetic_pdfs(root: Path) -> tuple[DatasetExample, ...]:
    """Read the synthetic set written by the generator. Order follows nfe id."""
    examples: list[DatasetExample] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        planted = metadata.get("planted_error")
        rendition = metadata.get("rendition") or DIGITAL_PDF
        examples.append(
            DatasetExample(
                nfe_id=metadata["nfe_id"],
                origin=metadata["origin"],
                pdf=(folder / "danfe.pdf").read_bytes(),
                ground_truth=Extraction.model_validate_json(
                    (folder / "ground_truth.json").read_text(encoding="utf-8")
                ),
                planted_rule=None if planted is None else str(planted["rule"]),
                rendition=str(rendition),
            )
        )
    return tuple(examples)
