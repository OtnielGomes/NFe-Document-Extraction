"""LangSmith dataset uploads. The repository stays the source of truth."""

import json
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_identifier_rules import _extraction, _line_item

from nfe_document_extraction.domain import Extraction
from nfe_document_extraction.evaluation.dataset import load_synthetic_pdfs
from nfe_document_extraction.evaluation.langsmith import (
    DATASET_DESCRIPTION,
    DATASET_NAME,
    DatasetUpload,
    ExperimentExample,
    LangSmithDatasetStore,
    LangSmithExperimentPublisher,
    ScoredReading,
    example_uploads,
    repository_commit,
    score_extraction,
    sync_dataset,
    upload_experiments,
)
from nfe_document_extraction.evaluation.scoring import score_document
from nfe_document_extraction.evaluation.tracing import disable_tracing

_METADATA_KEYS = {
    "origin",
    "input_type",
    "nfe_id",
    "has_icms_st",
    "planted_error",
}


def test_sync_payload_has_one_example_per_rendition_with_splits(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    uploads = example_uploads(load_synthetic_pdfs(dataset))

    by_rendition = {
        (item.inputs["nfe_id"], item.inputs["input_type"]): item for item in uploads
    }
    assert len(uploads) == len(by_rendition) == 3

    digital = by_rendition[("synthetic-001", "digital PDF")]
    photo = by_rendition[("synthetic-001", "phone photo")]
    planted = by_rendition[("synthetic-021", "digital PDF")]

    assert digital.example_id != photo.example_id
    assert set(digital.metadata) == _METADATA_KEYS
    assert digital.metadata["origin"] == "synthetic"
    assert digital.metadata["input_type"] == "digital PDF"
    assert digital.metadata["nfe_id"] == "synthetic-001"
    assert digital.metadata["has_icms_st"] is True
    assert digital.metadata["planted_error"] is None
    assert digital.split == "clean"
    assert digital.outputs["ground_truth"]["header"]["model"] == "55"

    assert photo.split == "clean"
    assert photo.metadata["input_type"] == "phone photo"
    assert photo.metadata["planted_error"] is None

    assert planted.split == "planted_errors"
    assert planted.metadata["has_icms_st"] is False
    assert planted.metadata["planted_error"] == {
        "rule": "quantity_times_unit_price",
        "field": "line_items[0].unit_price",
    }


def test_sync_creates_updates_and_drops_renditions_without_churn(
    tmp_path: Path,
) -> None:
    store = _MemoryStore()
    documents = load_synthetic_pdfs(_dataset(tmp_path))

    created = sync_dataset(store, documents)

    assert store.description == DATASET_DESCRIPTION
    assert len(store.list_examples(DATASET_NAME)) == 3
    assert created
    assert sync_dataset(store, documents) == created

    updated = replace(documents[0], has_icms_st=False)
    changed = sync_dataset(store, (updated, *documents[1:-1]))

    assert changed != created
    stored = {item.example_id: item for item in store.list_examples(DATASET_NAME)}
    assert len(stored) == 2
    assert (
        stored[example_uploads((updated,))[0].example_id].metadata["has_icms_st"]
        is False
    )
    assert example_uploads((documents[-1],))[0].example_id not in stored


def test_langsmith_client_stores_split_and_metadata(tmp_path: Path) -> None:
    documents = load_synthetic_pdfs(_dataset(tmp_path))
    client = _FakeLangSmith()

    version = sync_dataset(LangSmithDatasetStore(client), documents)

    assert version == "2026-10-05T12:00:00+00:00"
    assert client.created == {
        "name": DATASET_NAME,
        "description": DATASET_DESCRIPTION,
    }
    assert {item["split"] for item in client.created_examples} == {
        "clean",
        "planted_errors",
    }
    planted = next(
        item for item in client.created_examples if item["split"] == "planted_errors"
    )
    assert planted["metadata"] == {
        "origin": "synthetic",
        "input_type": "digital PDF",
        "nfe_id": "synthetic-021",
        "has_icms_st": False,
        "planted_error": {
            "rule": "quantity_times_unit_price",
            "field": "line_items[0].unit_price",
        },
    }

    client.created_examples.clear()
    updated = replace(documents[0], has_icms_st=False)
    sync_dataset(LangSmithDatasetStore(client), (updated, *documents[1:]))

    assert client.created_examples == []
    assert len(client.updated_examples) == 1
    assert client.updated_examples[0]["metadata"]["has_icms_st"] is False
    assert client.updated_examples[0]["split"] == "clean"


def test_evaluator_matches_the_local_score_for_one_wrong_total() -> None:
    invoice = _extraction()
    reading = _extraction(invoice_total=1_181)
    local = score_document(reading, invoice)
    feedback = _feedback(
        score_extraction(
            _prediction(reading),
            {"ground_truth": invoice.model_dump(mode="json")},
        )
    )

    assert feedback["field_correct"]["score"] == 28
    assert feedback["field_opportunities"]["score"] == 29
    assert feedback["perfect_invoice"]["score"] == 0
    assert feedback["missing_items"]["score"] == 0
    assert feedback["spurious_items"]["score"] == 0
    assert feedback["description_similarity"]["value"] == "1"
    assert feedback["field_correct"]["score"] == local.field_correct
    assert feedback["field_opportunities"]["score"] == local.field_opportunities
    assert feedback["perfect_invoice"]["score"] == int(local.perfect)
    assert feedback["missing_items"]["score"] == local.missing_items
    assert feedback["spurious_items"]["score"] == local.spurious_items


def test_evaluator_counts_one_missing_item() -> None:
    first = _line_item(product_code="A", description="Alpha", item_total=100)
    second = _line_item(
        printed_order=2,
        product_code="B",
        description="Beta",
        item_total=200,
    )
    third = _line_item(
        printed_order=3,
        product_code="C",
        description="Gamma",
        item_total=300,
    )
    invoice = _extraction(line_items=(first, second, third))
    reading = _extraction(line_items=(first, third))
    local = score_document(reading, invoice)
    feedback = _feedback(
        score_extraction(
            _prediction(reading),
            {"ground_truth": invoice.model_dump(mode="json")},
        )
    )

    assert feedback["missing_items"]["score"] == 1
    assert feedback["spurious_items"]["score"] == 0
    assert feedback["field_correct"]["score"] == 37
    assert feedback["field_opportunities"]["score"] == 38
    assert feedback["missing_items"]["score"] == local.missing_items
    assert feedback["field_correct"]["score"] == local.field_correct


def test_evaluator_scores_nothing_when_extraction_failed() -> None:
    invoice = _extraction()
    feedback = _feedback(
        score_extraction(
            {"extraction": None, "extraction_failure": "parse_failure"},
            {"ground_truth": invoice.model_dump(mode="json")},
        )
    )

    assert feedback["field_correct"]["score"] is None
    assert feedback["perfect_invoice"]["score"] is None
    assert feedback["description_similarity"]["score"] is None


def test_experiment_records_commit_dataset_version_and_local_scores(
    tmp_path: Path,
) -> None:
    documents = load_synthetic_pdfs(_dataset(tmp_path))
    invoice = documents[0].ground_truth
    reading = _extraction(invoice_total=1_181)
    local = score_document(reading, invoice)
    publisher = _RecordingPublisher()

    upload_experiments(
        publisher,
        (
            ScoredReading(
                approach="cheap vision",
                nfe_id=documents[0].nfe_id,
                input_type=documents[0].input_type,
                outputs=_prediction(reading),
            ),
            ScoredReading(
                approach="strong vision",
                nfe_id=documents[0].nfe_id,
                input_type=documents[0].input_type,
                outputs=_prediction(invoice),
            ),
        ),
        documents,
        commit="abc123",
        dataset_version="2026-10-05T12:00:00+00:00",
    )

    assert [item["approach"] for item in publisher.experiments] == [
        "cheap vision",
        "strong vision",
    ]
    assert {item["commit"] for item in publisher.experiments} == {"abc123"}
    assert {item["dataset_version"] for item in publisher.experiments} == {
        "2026-10-05T12:00:00+00:00"
    }
    assert {item["dataset_name"] for item in publisher.experiments} == {DATASET_NAME}
    assert all(item["evaluator"] is score_extraction for item in publisher.experiments)
    cheap = publisher.experiments[0]
    assert len(cheap["rows"]) == 1
    feedback = _feedback(cheap["rows"][0])
    assert feedback["field_correct"]["score"] == 28
    assert feedback["field_correct"]["score"] == local.field_correct
    perfect = _feedback(publisher.experiments[1]["rows"][0])
    assert perfect["perfect_invoice"]["score"] == 1
    assert perfect["field_correct"]["score"] == perfect["field_opportunities"]["score"]


def test_langsmith_publisher_uploads_the_commit_unless_tracing_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nfe_document_extraction.evaluation import tracing

    tracing._forced_off.set(False)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    client = _EvaluateClient()
    dataset_id = uuid.uuid4()
    example = ExperimentExample(
        example_id="6f0c9a3e-1b7d-4c2a-8e5f-9a1b2c3d4e5f",
        inputs={"nfe_id": "synthetic-001", "input_type": "digital PDF"},
        reference_outputs={"ground_truth": {"header": {"model": "55"}}},
    )
    publisher = LangSmithExperimentPublisher(client, dataset_id)

    publisher.publish(
        dataset_name=DATASET_NAME,
        approach="cheap vision",
        commit="abc123",
        dataset_version="2026-10-05T12:00:00+00:00",
        target=lambda inputs: {
            "extraction": None,
            "extraction_failure": "parse_failure",
        },
        evaluator=score_extraction,
        examples=(example,),
    )

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["metadata"] == {
        "commit": "abc123",
        "dataset_version": "2026-10-05T12:00:00+00:00",
        "approach": "cheap vision",
        "dataset": DATASET_NAME,
    }
    assert call["evaluators"] == [score_extraction]
    assert call["experiment_prefix"] == "nfe-cheap-vision"
    assert call["max_concurrency"] == 1
    uploaded = call["data"][0]
    assert uploaded.id == uuid.UUID(example.example_id)
    assert uploaded.dataset_id == dataset_id
    assert uploaded.outputs == example.reference_outputs

    disable_tracing()
    publisher.publish(
        dataset_name=DATASET_NAME,
        approach="cheap vision",
        commit="abc123",
        dataset_version="2026-10-05T12:00:00+00:00",
        target=lambda inputs: {"extraction": None},
        evaluator=score_extraction,
        examples=(example,),
    )

    assert len(client.calls) == 1


def test_repository_commit_is_the_checked_out_revision() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()

    assert repository_commit(root) == expected


class _EvaluateClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def evaluate(
        self,
        target: object,
        /,
        data: object = None,
        evaluators: object = None,
        metadata: object = None,
        experiment_prefix: object = None,
        max_concurrency: object = None,
    ) -> None:
        self.calls.append(
            {
                "data": list(data),  # type: ignore[arg-type]
                "evaluators": evaluators,
                "metadata": metadata,
                "experiment_prefix": experiment_prefix,
                "max_concurrency": max_concurrency,
            }
        )


class _RecordingPublisher:
    def __init__(self) -> None:
        self.experiments: list[dict[str, object]] = []

    def publish(
        self,
        *,
        dataset_name: str,
        approach: str,
        commit: str,
        dataset_version: str,
        target: object,
        evaluator: object,
        examples: Sequence[object],
    ) -> None:
        rows = []
        for example in examples:
            outputs = target(example.inputs)  # type: ignore[operator]
            rows.append(evaluator(outputs, example.reference_outputs))  # type: ignore[operator]
        self.experiments.append(
            {
                "dataset_name": dataset_name,
                "approach": approach,
                "commit": commit,
                "dataset_version": dataset_version,
                "evaluator": evaluator,
                "rows": rows,
            }
        )


def _feedback(result: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = result["results"]
    assert isinstance(rows, list)
    return {str(row["key"]): row for row in rows}


def _prediction(reading: Extraction) -> dict[str, object]:
    return {
        "extraction": reading.model_dump(mode="json"),
        "extraction_failure": None,
    }


class _MemoryStore:
    def __init__(self) -> None:
        self._datasets: dict[str, dict[str, DatasetUpload]] = {}
        self._versions: dict[str, int] = {}
        self.description: str | None = None

    def has_dataset(self, name: str) -> bool:
        return name in self._datasets

    def create_dataset(self, name: str, *, description: str) -> None:
        self._datasets[name] = {}
        self._versions[name] = 1
        self.description = description

    def list_examples(self, name: str) -> tuple[DatasetUpload, ...]:
        return tuple(self._datasets[name].values())

    def create_examples(self, name: str, examples: Sequence[DatasetUpload]) -> None:
        for example in examples:
            self._datasets[name][example.example_id] = example
        self._versions[name] += 1

    def update_examples(self, name: str, examples: Sequence[DatasetUpload]) -> None:
        for example in examples:
            self._datasets[name][example.example_id] = example
        self._versions[name] += 1

    def delete_examples(self, example_ids: Sequence[str]) -> None:
        for dataset in self._datasets.values():
            for example_id in example_ids:
                dataset.pop(example_id, None)
        for name in self._versions:
            self._versions[name] += 1

    def dataset_version(self, name: str) -> str:
        return f"v{self._versions[name]}"


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "synthetic"
    _write_invoice(
        root / "synthetic-001",
        nfe_id="synthetic-001",
        has_icms_st=True,
        split="clean",
        planted_error=None,
        renditions=("digital PDF", "phone photo"),
    )
    _write_invoice(
        root / "synthetic-021",
        nfe_id="synthetic-021",
        has_icms_st=False,
        split="planted_errors",
        planted_error={
            "rule": "quantity_times_unit_price",
            "field": "line_items[0].unit_price",
        },
        renditions=("digital PDF",),
    )
    return root


def _write_invoice(
    folder: Path,
    *,
    nfe_id: str,
    has_icms_st: bool,
    split: str,
    planted_error: dict[str, str] | None,
    renditions: tuple[str, ...],
) -> None:
    folder.mkdir(parents=True)
    metadata: dict[str, object] = {
        "nfe_id": nfe_id,
        "has_icms_st": has_icms_st,
        "origin": "synthetic",
        "split": split,
    }
    if planted_error is not None:
        metadata["planted_error"] = planted_error
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (folder / "ground_truth.json").write_text(
        _extraction().model_dump_json(),
        encoding="utf-8",
    )
    (folder / "danfe.pdf").write_bytes(b"%PDF-1.4")
    (folder / "renditions.json").write_text(
        json.dumps([{"input_type": rendition} for rendition in renditions]),
        encoding="utf-8",
    )


@dataclass
class _RemoteExample:
    id: str
    inputs: dict[str, str]
    outputs: dict[str, object]
    metadata: dict[str, object]
    split: str


class _FakeLangSmith:
    def __init__(self) -> None:
        self.created: dict[str, str] | None = None
        self.created_examples: list[dict[str, object]] = []
        self.updated_examples: list[dict[str, object]] = []
        self._remote: list[_RemoteExample] = []

    def has_dataset(self, *, dataset_name: str) -> bool:
        return self.created is not None or bool(self._remote)

    def create_dataset(self, dataset_name: str, *, description: str) -> None:
        self.created = {"name": dataset_name, "description": description}

    def list_examples(
        self,
        *,
        dataset_name: str,
        splits: list[str] | None = None,
    ) -> list[_RemoteExample]:
        if splits is None:
            return list(self._remote)
        return [item for item in self._remote if item.split in splits]

    def create_examples(
        self,
        *,
        dataset_name: str,
        examples: list[dict[str, object]],
    ) -> None:
        self.created_examples.extend(examples)
        for example in examples:
            self._remote.append(
                _RemoteExample(
                    id=str(example["id"]),
                    inputs=dict(example["inputs"]),  # type: ignore[arg-type]
                    outputs=dict(example["outputs"]),  # type: ignore[arg-type]
                    metadata=dict(example["metadata"]),  # type: ignore[arg-type]
                    split=str(example["split"]),
                )
            )

    def update_examples(
        self,
        *,
        dataset_name: str,
        updates: list[dict[str, object]],
    ) -> None:
        self.updated_examples.extend(updates)
        by_id = {item.id: item for item in self._remote}
        for update in updates:
            current = by_id[str(update["id"])]
            current.inputs = dict(update["inputs"])  # type: ignore[arg-type]
            current.outputs = dict(update["outputs"])  # type: ignore[arg-type]
            current.metadata = dict(update["metadata"])  # type: ignore[arg-type]
            current.split = str(update["split"])

    def delete_examples(self, example_ids: Sequence[str]) -> None:
        removed = set(example_ids)
        self._remote = [item for item in self._remote if item.id not in removed]

    def read_dataset(self, *, dataset_name: str) -> object:
        return _DatasetVersion(datetime(2026, 10, 5, 12, 0, tzinfo=UTC))


@dataclass(frozen=True)
class _DatasetVersion:
    modified_at: datetime
    created_at: datetime | None = None
