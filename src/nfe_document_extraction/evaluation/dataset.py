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
SIMULATED_PHOTO_DIRECTORY = "simulated-photo"


@dataclass(frozen=True)
class DatasetExample:
    """One Rendition of a DANFE, its printed ground truth, and any planted error."""

    nfe_id: str
    origin: str
    pdf: bytes
    ground_truth: Extraction
    planted_rule: str | None
    has_icms_st: bool
    split: str
    planted_field: str | None
    rendition: str = DIGITAL_PDF
    pages: tuple[bytes, ...] = ()

    @property
    def input_type(self) -> str:
        """Rendition metadata recorded on the dataset example."""
        return self.rendition


def load_synthetic_pdfs(root: Path) -> tuple[DatasetExample, ...]:
    """Read the synthetic set. One example per Rendition, ordered by NF-e id.

    A folder with ``renditions.json`` yields one example per ``input_type``.
    Digital PDF and Simulated photo examples share that folder's ground truth.
    A folder without the manifest is a single Rendition, which is how a photo-only
    fixture is represented.
    """
    examples: list[DatasetExample] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        planted_rule, planted_field = _planted(metadata)
        ground_truth = Extraction.model_validate_json(
            (folder / "ground_truth.json").read_text(encoding="utf-8")
        )
        manifest_path = folder / "renditions.json"
        has_manifest = manifest_path.is_file()
        if has_manifest:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            input_types = tuple(str(entry["input_type"]) for entry in manifest)
        else:
            explicit = metadata.get("input_type") or metadata.get("rendition")
            input_types = (str(explicit or DIGITAL_PDF),)
        examples.extend(
            _example(
                folder,
                metadata,
                ground_truth,
                planted_rule,
                planted_field,
                input_type,
                require_photo_pages=has_manifest,
            )
            for input_type in input_types
        )
    return tuple(examples)


def _example(
    folder: Path,
    metadata: dict[str, object],
    ground_truth: Extraction,
    planted_rule: str | None,
    planted_field: str | None,
    input_type: str,
    *,
    require_photo_pages: bool,
) -> DatasetExample:
    pages = _simulated_photo_pages(folder) if input_type == SIMULATED_PHOTO else ()
    if input_type == SIMULATED_PHOTO and not pages and require_photo_pages:
        raise FileNotFoundError(f"{folder.name} is missing simulated photo pages")
    if pages:
        pdf = b""
    else:
        pdf = (folder / "danfe.pdf").read_bytes()
    return DatasetExample(
        nfe_id=str(metadata["nfe_id"]),
        origin=str(metadata["origin"]),
        pdf=pdf,
        ground_truth=ground_truth,
        planted_rule=planted_rule,
        has_icms_st=_bool_flag(metadata, "has_icms_st"),
        split=_split(metadata),
        planted_field=planted_field,
        rendition=input_type,
        pages=pages,
    )


_SPLITS = frozenset({"clean", "planted_errors"})


def _planted(metadata: dict[str, object]) -> tuple[str | None, str | None]:
    planted = metadata.get("planted_error")
    if planted is None:
        return None, None
    if not isinstance(planted, dict) or "rule" not in planted or "field" not in planted:
        raise ValueError(
            f"{metadata.get('nfe_id')} planted_error must include rule and field"
        )
    return str(planted["rule"]), str(planted["field"])


def _bool_flag(metadata: dict[str, object], name: str) -> bool:
    value = metadata[name]
    if not isinstance(value, bool):
        raise TypeError(f"{metadata.get('nfe_id')} {name} must be a boolean")
    return value


def _split(metadata: dict[str, object]) -> str:
    split = str(metadata["split"])
    if split not in _SPLITS:
        raise ValueError(f"{metadata.get('nfe_id')} has unknown split {split}")
    return split


def _simulated_photo_pages(folder: Path) -> tuple[bytes, ...]:
    photo_dir = folder / SIMULATED_PHOTO_DIRECTORY
    if not photo_dir.is_dir():
        return ()
    return tuple(path.read_bytes() for path in sorted(photo_dir.glob("page-*.jpg")))
