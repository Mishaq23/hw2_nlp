"""Prepare TinySMS partitions without question leakage."""

import hashlib

from datasets import Dataset, DatasetDict

DATASET_ID = "SlavaYus/tinysms_qa_12m"
DATASET_REVISION = "b2724689ff079e9c22efd548fe134bfb29036dbb"


def question_key(question: str) -> str:
    """Group case/whitespace variants without modifying the model input."""
    return " ".join(question.split()).casefold()


def prepare_raw_splits(
    raw: DatasetDict,
    validation_fraction: float = 0.02,
    seed: int = 42,
    num_proc: int | None = None,
    validation_data: Dataset | None = None,
) -> DatasetDict:
    """Filter missing training texts and hold out whole question groups.

    A stable hash assigns the same question to the same partition even when
    more shards are added. Official test questions are excluded from training
    and validation. Test order and original row IDs are preserved.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if "train" not in raw or "test" not in raw:
        raise ValueError("Expected train and test partitions.")
    if not {"question", "code"}.issubset(raw["train"].column_names):
        raise ValueError("train must contain question and code columns.")
    if "question" not in raw["test"].column_names:
        raise ValueError("test must contain a question column.")
    if any(not _nonempty_text(q) for q in raw["test"]["question"]):
        raise ValueError("Test contains missing questions; do not silently drop rows.")

    heldout_questions = {question_key(q) for q in raw["test"]["question"]}
    if validation_data is not None:
        if not {"question", "code"}.issubset(validation_data.column_names):
            raise ValueError("Fixed validation must contain question and code.")
        if any(
            not _nonempty_text(row["question"])
            or not _nonempty_text(row["code"])
            or question_key(row["question"]) in heldout_questions
            for row in validation_data
        ):
            raise ValueError(
                "Fixed validation contains invalid rows or official test questions."
            )
        heldout_questions.update(question_key(q) for q in validation_data["question"])
    train = raw["train"].map(
        lambda _, indices: {"sample_id": indices},
        batched=True,
        with_indices=True,
        num_proc=num_proc,
        desc="Assign original training row IDs",
    )
    train = train.filter(
        lambda questions, codes: [
            _nonempty_text(question)
            and _nonempty_text(code)
            and question_key(question) not in heldout_questions
            for question, code in zip(questions, codes)
        ],
        input_columns=["question", "code"],
        batched=True,
        num_proc=num_proc,
        desc="Remove missing texts and official test questions",
    )

    # Avoid Python's hash(): it is salted differently in each process.
    threshold = int(validation_fraction * (1 << 64))

    def assign_partition(batch):
        validation_flags = []
        for question in batch["question"]:
            digest = hashlib.blake2b(
                f"{seed}\0{question_key(question)}".encode("utf-8"), digest_size=8
            ).digest()
            validation_flags.append(int.from_bytes(digest, "big") < threshold)
        return {"_validation": validation_flags}

    grouped = train.map(
        assign_partition,
        batched=True,
        num_proc=num_proc,
        desc="Assign question groups to train/validation",
    )
    partitions = {}
    for name, is_validation in (("train", False), ("val", True)):
        if name == "val" and validation_data is not None:
            partitions[name] = validation_data
            if "sample_id" not in validation_data.column_names:
                partitions[name] = validation_data.add_column(
                    "sample_id", list(range(len(validation_data)))
                )
        else:
            partitions[name] = grouped.filter(
                lambda flags: [flag == is_validation for flag in flags],
                input_columns=["_validation"],
                batched=True,
                num_proc=num_proc,
                desc=f"Select {name}",
            ).remove_columns("_validation")
        if not len(partitions[name]):
            raise ValueError(
                f"{name} is empty. Use more examples or another validation_fraction."
            )

    test = raw["test"].select_columns(["question"])
    partitions["test"] = test.map(
        lambda _, indices: {"sample_id": indices},
        batched=True,
        with_indices=True,
        desc="Assign original test row IDs",
    )
    return DatasetDict(partitions)


def _nonempty_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())
