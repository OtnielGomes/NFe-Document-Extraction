"""Ground truth for a real DANFE is the value printed, derived from its XML."""

from pathlib import Path

from nfe_document_extraction.domain import Extraction
from nfe_document_extraction.evaluation.real_set import ground_truth_from_xml

_SYNTHETIC = Path(__file__).resolve().parents[1] / "data" / "synthetic"


def test_ground_truth_matches_the_value_printed_on_every_synthetic_danfe() -> None:
    folders = sorted(path for path in _SYNTHETIC.iterdir() if path.is_dir())

    assert len(folders) == 100
    for folder in folders:
        derived = ground_truth_from_xml(
            (folder / "nfe.xml").read_text(encoding="utf-8")
        )
        printed = Extraction.model_validate_json(
            (folder / "ground_truth.json").read_text(encoding="utf-8")
        )
        assert derived == printed, folder.name


def test_nonzero_unsupported_totals_are_present_on_the_danfe() -> None:
    xml = (_SYNTHETIC / "synthetic-001" / "nfe.xml").read_text(encoding="utf-8")
    xml = xml.replace("<vII>0.00</vII>", "<vII>1.00</vII>")
    xml = xml.replace("<vIPIDevol>0.00</vIPIDevol>", "<vIPIDevol>2.00</vIPIDevol>")
    xml = xml.replace("<vICMSDeson>0.00</vICMSDeson>", "<vICMSDeson>3.00</vICMSDeson>")
    xml = xml.replace("<vFCPST>0.00</vFCPST>", "<vFCPST>4.00</vFCPST>")

    flags = ground_truth_from_xml(xml).header.unsupported_components

    assert flags.import_tax is True
    assert flags.returned_ipi is True
    assert flags.icms_relief is True
    assert flags.fcp_st is True
