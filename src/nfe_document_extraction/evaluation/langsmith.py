"""Upload the repository dataset to LangSmith and score experiments with it.

The repository is the source of truth. LangSmith receives one example per
Rendition; the files themselves stay here.
"""

from __future__ import annotations

import subprocess
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from nfe_document_extraction.domain import Extraction
from nfe_document_extraction.evaluation.dataset import DatasetExample
from nfe_document_extraction.evaluation.scoring import score_document
from nfe_document_extraction.evaluation.tracing import send_to_langsmith

_EXAMPLE_NAMESPACE = uuid.UUID("6f0c9a3e-1b7d-4c2a-8e5f-9a1b2c3d4e5f")
DATASET_NAME = "nfe-synthetic"
DATASET_DESCRIPTION = (
    "DANFE renditions synced from the repository, which stays the source of truth."
)
_METADATA_KEYS = ("origin", "input_type", "nfe_id", "has_icms_st", "planted_error")
_CLEAN_SPLIT = "clean"
_PLANTED_SPLIT = "planted_errors"


@dataclass(frozen=True)
class DatasetUpload:
    """One Rendition ready to create or update in the LangSmith dataset."""

    example_id: str
    inputs: dict[str, str]
    outputs: dict[str, Any]
    metadata: dict[str, Any]
    split: str


class DatasetStore(Protocol):
    """The LangSmith dataset operations sync needs. Tests use an in-memory store."""

    def has_dataset(self, name: str) -> bool: ...

    def create_dataset(self, name: str, *, description: str) -> None: ...

    def list_examples(self, name: str) -> tuple[DatasetUpload, ...]: ...

    def create_examples(self, name: str, examples: Sequence[DatasetUpload]) -> None: ...

    def update_examples(self, name: str, examples: Sequence[DatasetUpload]) -> None: ...

    def delete_examples(self, example_ids: Sequence[str]) -> None: ...

    def dataset_version(self, name: str) -> str: ...


def sync_dataset(
    store: DatasetStore,
    documents: Sequence[DatasetExample],
    *,
    name: str = DATASET_NAME,
) -> str:
    """Create or update the dataset from the repository. Returns its version."""
    uploads = example_uploads(documents)
    if not store.has_dataset(name):
        store.create_dataset(name, description=DATASET_DESCRIPTION)
    existing = {item.example_id: item for item in store.list_examples(name)}
    incoming = {item.example_id: item for item in uploads}
    to_create = [item for item in uploads if item.example_id not in existing]
    to_update = [
        item
        for item in uploads
        if item.example_id in existing and existing[item.example_id] != item
    ]
    to_delete = [example_id for example_id in existing if example_id not in incoming]
    if to_create:
        store.create_examples(name, to_create)
    if to_update:
        store.update_examples(name, to_update)
    if to_delete:
        store.delete_examples(to_delete)
    return store.dataset_version(name)


class DatasetNotSynced(Exception):
    """The LangSmith dataset has not been created from the repository yet."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class _RemoteExample(Protocol):
    id: object
    inputs: Mapping[str, Any] | None
    outputs: Mapping[str, Any] | None
    metadata: Mapping[str, Any] | None


class _DatasetVersion(Protocol):
    id: object
    modified_at: datetime | None
    created_at: datetime


class LangSmithClient(Protocol):
    def has_dataset(self, *, dataset_name: str) -> bool: ...

    def create_dataset(self, dataset_name: str, *, description: str) -> object: ...

    def list_examples(
        self,
        *,
        dataset_name: str,
        splits: Sequence[str] | None = None,
    ) -> Iterable[_RemoteExample]: ...

    def create_examples(
        self,
        *,
        dataset_name: str,
        examples: Sequence[Mapping[str, Any]],
    ) -> object: ...

    def update_examples(
        self,
        *,
        dataset_name: str,
        updates: Sequence[Mapping[str, Any]],
    ) -> object: ...

    def delete_examples(self, example_ids: Sequence[str]) -> None: ...

    def read_dataset(self, *, dataset_name: str) -> _DatasetVersion: ...


class LangSmithDatasetStore:
    """Adapt the LangSmith client to the dataset sync store."""

    def __init__(self, client: LangSmithClient) -> None:
        self._client = client

    def has_dataset(self, name: str) -> bool:
        return self._client.has_dataset(dataset_name=name)

    def create_dataset(self, name: str, *, description: str) -> None:
        self._client.create_dataset(name, description=description)

    def list_examples(self, name: str) -> tuple[DatasetUpload, ...]:
        clean = self._ids(name, _CLEAN_SPLIT)
        planted = self._ids(name, _PLANTED_SPLIT)
        uploads: list[DatasetUpload] = []
        for example in self._client.list_examples(dataset_name=name):
            example_id = str(example.id)
            if example_id in clean:
                split = _CLEAN_SPLIT
            elif example_id in planted:
                split = _PLANTED_SPLIT
            else:
                split = ""
            uploads.append(
                DatasetUpload(
                    example_id=example_id,
                    inputs={
                        str(key): str(value)
                        for key, value in (example.inputs or {}).items()
                    },
                    outputs=dict(example.outputs or {}),
                    metadata={
                        key: (example.metadata or {}).get(key) for key in _METADATA_KEYS
                    },
                    split=split,
                )
            )
        return tuple(uploads)

    def create_examples(self, name: str, examples: Sequence[DatasetUpload]) -> None:
        self._client.create_examples(
            dataset_name=name,
            examples=[_payload(item) for item in examples],
        )

    def update_examples(self, name: str, examples: Sequence[DatasetUpload]) -> None:
        self._client.update_examples(
            dataset_name=name,
            updates=[_payload(item) for item in examples],
        )

    def delete_examples(self, example_ids: Sequence[str]) -> None:
        self._client.delete_examples(list(example_ids))

    def dataset_version(self, name: str) -> str:
        if not self.has_dataset(name):
            raise DatasetNotSynced(name)
        dataset = self._client.read_dataset(dataset_name=name)
        stamp = dataset.modified_at or dataset.created_at
        return stamp.isoformat()

    def dataset_id(self, name: str) -> uuid.UUID:
        if not self.has_dataset(name):
            raise DatasetNotSynced(name)
        return uuid.UUID(str(self._client.read_dataset(dataset_name=name).id))

    def _ids(self, name: str, split: str) -> set[str]:
        return {
            str(example.id)
            for example in self._client.list_examples(dataset_name=name, splits=[split])
        }


def prediction_outputs(
    extraction: Extraction | None,
    failure: str | None,
) -> dict[str, Any]:
    """The experiment output for one reading. A failure carries no extraction."""
    if extraction is None or failure is not None:
        return {"extraction": None, "extraction_failure": failure}
    return {
        "extraction": extraction.model_dump(mode="json"),
        "extraction_failure": None,
    }


def repository_commit(root: Path) -> str:
    """The git commit this experiment was produced from."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or not commit:
        raise RuntimeError("could not read the git commit for this experiment")
    return commit


def upload_recorded_run(
    client: LangSmithClient,
    readings: Sequence[ScoredReading],
    documents: Sequence[DatasetExample],
    *,
    commit: str,
    dataset_name: str = DATASET_NAME,
) -> None:
    """Upload a finished local run. Does not contact LangSmith when tracing is off."""

    def send() -> None:
        store = LangSmithDatasetStore(client)
        version = store.dataset_version(dataset_name)
        upload_experiments(
            LangSmithExperimentPublisher(client, store.dataset_id(dataset_name)),
            readings,
            documents,
            commit=commit,
            dataset_version=version,
            dataset_name=dataset_name,
        )

    send_to_langsmith(send)


def score_extraction(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    """Score one prediction with the same function the local eval uses.

    LangSmith calls this with the run outputs and the dataset reference.
    A failed extraction has no field score, matching a local score of ``None``.
    """
    extraction = outputs.get("extraction")
    failure = outputs.get("extraction_failure")
    if extraction is None or failure:
        return {"results": [_blank(key) for key in _SCORE_KEYS]}
    score = score_document(
        Extraction.model_validate(extraction),
        Extraction.model_validate(reference_outputs["ground_truth"]),
    )
    similarities = score.description_similarities
    if similarities:
        mean = sum(similarities, Decimal(0)) / Decimal(len(similarities))
        similarity: dict[str, Any] = {
            "key": "description_similarity",
            "score": float(mean),
            "value": format(mean, "f"),
        }
    else:
        similarity = _blank("description_similarity")
    return {
        "results": [
            {"key": "field_correct", "score": score.field_correct},
            {"key": "field_opportunities", "score": score.field_opportunities},
            {"key": "missing_items", "score": score.missing_items},
            {"key": "spurious_items", "score": score.spurious_items},
            {"key": "perfect_invoice", "score": int(score.perfect)},
            similarity,
        ]
    }


_SCORE_KEYS = (
    "field_correct",
    "field_opportunities",
    "missing_items",
    "spurious_items",
    "perfect_invoice",
    "description_similarity",
)


def _blank(key: str) -> dict[str, Any]:
    return {"key": key, "score": None}


@dataclass(frozen=True)
class ScoredReading:
    """One approach's prediction for one Rendition."""

    approach: str
    nfe_id: str
    input_type: str
    outputs: dict[str, Any]


@dataclass(frozen=True)
class ExperimentExample:
    """A dataset row the experiment publisher scores."""

    example_id: str
    inputs: dict[str, str]
    reference_outputs: dict[str, Any]


class ExperimentPublisher(Protocol):
    def publish(
        self,
        *,
        dataset_name: str,
        approach: str,
        commit: str,
        dataset_version: str,
        target: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        evaluator: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
        examples: Sequence[ExperimentExample],
    ) -> None: ...


def upload_experiments(
    publisher: ExperimentPublisher,
    readings: Sequence[ScoredReading],
    documents: Sequence[DatasetExample],
    *,
    commit: str,
    dataset_version: str,
    dataset_name: str = DATASET_NAME,
) -> None:
    """Upload one experiment per approach. Each records commit and dataset version."""
    references = {
        (document.nfe_id, document.input_type): {
            "ground_truth": document.ground_truth.model_dump(mode="json")
        }
        for document in documents
    }
    for approach in dict.fromkeys(reading.approach for reading in readings):
        group = [reading for reading in readings if reading.approach == approach]
        predictions = {
            (reading.nfe_id, reading.input_type): reading.outputs for reading in group
        }
        publisher.publish(
            dataset_name=dataset_name,
            approach=approach,
            commit=commit,
            dataset_version=dataset_version,
            target=_target(predictions),
            evaluator=score_extraction,
            examples=tuple(
                ExperimentExample(
                    example_id=_example_id(reading.nfe_id, reading.input_type),
                    inputs={
                        "nfe_id": reading.nfe_id,
                        "input_type": reading.input_type,
                    },
                    reference_outputs=references[(reading.nfe_id, reading.input_type)],
                )
                for reading in group
            ),
        )


class LangSmithExperimentPublisher:
    """Upload experiments through the LangSmith client, unless tracing is off."""

    def __init__(self, client: Any, dataset_id: uuid.UUID) -> None:
        self._client = client
        self._dataset_id = dataset_id

    def publish(
        self,
        *,
        dataset_name: str,
        approach: str,
        commit: str,
        dataset_version: str,
        target: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        evaluator: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
        examples: Sequence[ExperimentExample],
    ) -> None:
        if not examples:
            return
        from langsmith.schemas import Example

        rows = [
            Example(
                id=uuid.UUID(item.example_id),
                dataset_id=self._dataset_id,
                inputs=dict(item.inputs),
                outputs=dict(item.reference_outputs),
            )
            for item in examples
        ]

        def send() -> None:
            self._client.evaluate(
                target,
                data=rows,
                evaluators=[evaluator],
                metadata={
                    "commit": commit,
                    "dataset_version": dataset_version,
                    "approach": approach,
                    "dataset": dataset_name,
                },
                experiment_prefix=_experiment_prefix(approach),
                max_concurrency=1,
            )

        send_to_langsmith(send)


def _experiment_prefix(approach: str) -> str:
    slug = "-".join(part for part in approach.replace("+", " ").split() if part)
    return f"nfe-{slug}"


def _target(
    predictions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    def target(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        return predictions[(str(inputs["nfe_id"]), str(inputs["input_type"]))]

    return target


def _example_id(nfe_id: str, input_type: str) -> str:
    return str(uuid.uuid5(_EXAMPLE_NAMESPACE, f"{nfe_id}\n{input_type}"))


def _payload(item: DatasetUpload) -> dict[str, Any]:
    return {
        "id": item.example_id,
        "inputs": item.inputs,
        "outputs": item.outputs,
        "metadata": item.metadata,
        "split": item.split,
    }


def example_uploads(documents: Sequence[DatasetExample]) -> tuple[DatasetUpload, ...]:
    """Map each Rendition to a LangSmith example. Ids are stable across syncs."""
    uploads = tuple(_upload(document) for document in documents)
    ids = [item.example_id for item in uploads]
    if len(ids) != len(set(ids)):
        raise ValueError("two renditions resolved to the same LangSmith example id")
    return uploads


def _upload(document: DatasetExample) -> DatasetUpload:
    if document.planted_rule is None:
        planted: dict[str, str] | None = None
    elif document.planted_field is None:
        raise ValueError(f"{document.nfe_id} planted error is missing its field")
    else:
        planted = {"rule": document.planted_rule, "field": document.planted_field}
    return DatasetUpload(
        example_id=_example_id(document.nfe_id, document.input_type),
        inputs={"nfe_id": document.nfe_id, "input_type": document.input_type},
        outputs={"ground_truth": document.ground_truth.model_dump(mode="json")},
        metadata={
            "origin": document.origin,
            "input_type": document.input_type,
            "nfe_id": document.nfe_id,
            "has_icms_st": document.has_icms_st,
            "planted_error": planted,
        },
        split=document.split,
    )
