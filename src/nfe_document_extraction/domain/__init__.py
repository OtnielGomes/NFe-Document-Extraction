"""Schema and normalization for an Extraction."""

from nfe_document_extraction.domain.extraction import (
    Extraction,
    Header,
    LineItem,
    UnsupportedComponents,
)

__all__ = [
    "Extraction",
    "Header",
    "LineItem",
    "UnsupportedComponents",
]
