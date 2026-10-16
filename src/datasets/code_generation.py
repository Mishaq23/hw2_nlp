"""Cached, unpadded seq2seq examples; batching belongs to collate_fn."""

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from datasets import Dataset as ArrowDataset
from datasets import Features, Sequence, Value, load_from_disk
from src.datasets.tokenizer import CodeTokenizer


def dataset_digest(data):
    """Content identity for small fixed splits, stable across Arrow save/load."""
    digest = hashlib.sha256()
    for batch in data.iter(batch_size=256):
        digest.update(json.dumps(batch, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def tokenize_dataset(
    data: ArrowDataset,
    tokenizer: CodeTokenizer,
    has_targets: bool = True,
    batch_size: int = 1000,
    num_proc: int | None = None,
) -> ArrowDataset:
    """Cache full sequences in Arrow without padding or silent truncation.

    encoder = question + EOS
    decoder = BOS + code
    labels  = code + EOS

    Test examples only have encoder inputs. Sequence lengths include boundary
    tokens, so they can later be used by a sampler or an explicit length filter.
    """
    required = {"question", "sample_id"}
    if has_targets:
        required.add("code")
    if not required.issubset(data.column_names):
        raise ValueError(
            f"Missing required columns: {required - set(data.column_names)}"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    # Compact IDs on disk; __getitem__ converts them to torch.long for embedding.
    features = {
        "sample_id": Value("int64"),
        "encoder_input_ids": Sequence(Value("int32")),
        "source_length": Value("int32"),
    }
    if has_targets:
        features.update(
            decoder_input_ids=Sequence(Value("int32")),
            labels=Sequence(Value("int32")),
            target_length=Value("int32"),
        )

    def encode_batch(batch):
        questions = tokenizer.encode_batch(batch["question"])
        encoder_ids = [ids + [tokenizer.eos_id] for ids in questions]
        result = {
            "sample_id": batch["sample_id"],
            "encoder_input_ids": encoder_ids,
            "source_length": [len(ids) for ids in encoder_ids],
        }
        if has_targets:
            codes = tokenizer.encode_batch(batch["code"])
            result.update(
                decoder_input_ids=[[tokenizer.bos_id] + ids for ids in codes],
                labels=[ids + [tokenizer.eos_id] for ids in codes],
                target_length=[len(ids) + 1 for ids in codes],
            )
        return result

    return data.map(
        encode_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        features=Features(features),
        remove_columns=data.column_names,
        desc=(
            "Tokenize questions and code" if has_targets else "Tokenize test questions"
        ),
    )


class CodeGenerationDataset(Dataset):
    """Wrap tokenized Arrow data for the existing PyTorch project template.

    Returns variable-length CPU long tensors; collate handles padding.
    """

    def __init__(self, data: ArrowDataset | str | Path):
        if isinstance(data, (str, Path)):
            data = load_from_disk(str(data))
        if not isinstance(data, ArrowDataset):
            raise TypeError("Expected an Arrow Dataset or a saved split directory.")
        if not {"encoder_input_ids", "sample_id"}.issubset(data.column_names):
            raise ValueError("Pass data produced by tokenize_dataset().")
        target_columns = {"decoder_input_ids", "labels"}
        present_targets = target_columns.intersection(data.column_names)
        if present_targets and present_targets != target_columns:
            raise ValueError("decoder_input_ids and labels must be present together.")
        self.data = data
        self.has_targets = bool(present_targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        example = {
            "sample_id": row["sample_id"],
            "encoder_input_ids": torch.tensor(
                row["encoder_input_ids"], dtype=torch.long
            ),
        }
        if self.has_targets:
            example.update(
                decoder_input_ids=torch.tensor(
                    row["decoder_input_ids"], dtype=torch.long
                ),
                labels=torch.tensor(row["labels"], dtype=torch.long),
            )
        return example
