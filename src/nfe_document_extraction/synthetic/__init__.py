"""Reproducible fictitious NF-e, DANFE, and ground truth."""

from nfe_document_extraction.synthetic.generate import (
    SYNTHETIC_COUNT,
    SYNTHETIC_SEED,
    PlantedError,
    SyntheticDocument,
    generate_synthetic_set,
    write_synthetic_set,
)
from nfe_document_extraction.synthetic.photos import simulate_photo_pages

__all__ = [
    "SYNTHETIC_COUNT",
    "SYNTHETIC_SEED",
    "PlantedError",
    "SyntheticDocument",
    "generate_synthetic_set",
    "simulate_photo_pages",
    "write_synthetic_set",
]
