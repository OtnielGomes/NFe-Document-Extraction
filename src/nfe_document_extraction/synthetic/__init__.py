"""Reproducible fictitious NF-e, DANFE, and ground truth."""

from nfe_document_extraction.synthetic.generate import (
    SYNTHETIC_COUNT,
    SYNTHETIC_SEED,
    PlantedError,
    SyntheticDocument,
    generate_synthetic_set,
    write_synthetic_set,
)

__all__ = [
    "SYNTHETIC_COUNT",
    "SYNTHETIC_SEED",
    "PlantedError",
    "SyntheticDocument",
    "generate_synthetic_set",
    "write_synthetic_set",
]
