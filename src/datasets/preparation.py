"""Prepare TinySMS once and reuse the saved tokenizer and Arrow splits."""

import json
from copy import deepcopy
from pathlib import Path
from shutil import copy2

from omegaconf import OmegaConf

from datasets import DatasetDict, load_dataset, load_from_disk
from src.datasets.code_generation import dataset_digest, tokenize_dataset
from src.datasets.tinysms import prepare_raw_splits
from src.datasets.tokenizer import CodeTokenizer, tokenizer_digest


def _cache_settings(settings):
    """Locations may change when Kaggle mounts the previous session's output."""
    settings = deepcopy(settings)
    for key in ("output_dir", "pretrained_tokenizer_path", "validation_path"):
        settings.pop(key, None)
    settings["dataset"].pop("cache_dir", None)
    return settings


def prepare_data(config):
    settings = OmegaConf.to_container(config, resolve=True)
    if not settings.pop("enabled", True):
        return
    output = Path(settings["output_dir"])
    pretrained_path = settings.get("pretrained_tokenizer_path")
    fixed_validation = (
        load_from_disk(settings["validation_path"])
        if settings.get("validation_path")
        else None
    )
    inputs = {
        "tokenizer_sha256": (
            tokenizer_digest(pretrained_path) if pretrained_path else None
        ),
        "validation_fingerprint": (
            dataset_digest(fixed_validation) if fixed_validation is not None else None
        ),
    }
    metadata_path = output / "metadata.json"
    cached_files = [output / "tokenizer.json", metadata_path] + [
        output / "encoded" / split / "state.json" for split in ("train", "val", "test")
    ]
    if all(path.is_file() for path in cached_files):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached_digest = tokenizer_digest(output / "tokenizer.json")
        original_inputs = metadata.get("inputs", {})
        validation_matches = (
            original_inputs.get("validation_fingerprint")
            == inputs["validation_fingerprint"]
        )
        if (
            fixed_validation is not None
            and "encoder_input_ids" in fixed_validation.column_names
        ):
            saved_fingerprint = metadata.get("validation_sha256")
            if saved_fingerprint is None:
                saved_fingerprint = dataset_digest(
                    load_from_disk(str(output / "encoded" / "val"))
                )
            validation_matches = inputs["validation_fingerprint"] == saved_fingerprint
        tokenizer_matches = (
            inputs["tokenizer_sha256"] == cached_digest
            if pretrained_path
            else original_inputs.get("tokenizer_sha256") is None
        )
        if (
            _cache_settings(metadata["config"]) == _cache_settings(settings)
            and tokenizer_matches
            and validation_matches
            and metadata.get("tokenizer_sha256", cached_digest) == cached_digest
        ):
            print(f"Using prepared data: {output.resolve()}")
            return

    tokenizer = CodeTokenizer.from_file(pretrained_path) if pretrained_path else None
    encoded_validation = None
    if (
        fixed_validation is not None
        and "encoder_input_ids" in fixed_validation.column_names
    ):
        if tokenizer is None:
            raise ValueError("Encoded validation requires pretrained_tokenizer_path.")
        encoded_validation = fixed_validation
        fixed_validation = fixed_validation.map(
            lambda row: {
                "question": tokenizer.decode(row["encoder_input_ids"]),
                "code": tokenizer.decode(row["labels"]),
            },
            remove_columns=[
                name for name in fixed_validation.column_names if name != "sample_id"
            ],
            desc="Read fixed validation questions",
        )
    raw = load_dataset(**settings["dataset"])
    splits = prepare_raw_splits(
        raw, **settings["split"], validation_data=fixed_validation
    )
    training_rows = 0
    if tokenizer is None:
        tokenizer_train = splits["train"]
        limit = settings["tokenizer_train_rows"]
        if limit is not None:
            if limit < 1:
                raise ValueError("tokenizer_train_rows must be positive or null.")
            tokenizer_train = tokenizer_train.shuffle(
                seed=settings["split"]["seed"]
            ).select(range(min(limit, len(tokenizer_train))))
        training_rows = len(tokenizer_train)
        tokenizer = CodeTokenizer.train(tokenizer_train, **settings["tokenizer"])
    encoded = DatasetDict(
        {
            name: (
                encoded_validation
                if name == "val" and encoded_validation is not None
                else tokenize_dataset(
                    data, tokenizer, has_targets=name != "test", **settings["tokenize"]
                )
            )
            for name, data in splits.items()
        }
    )
    if pretrained_path:
        output.mkdir(parents=True, exist_ok=True)
        if Path(pretrained_path).resolve() != (output / "tokenizer.json").resolve():
            copy2(pretrained_path, output / "tokenizer.json")
    else:
        tokenizer.save(output / "tokenizer.json")
    encoded.save_to_disk(str(output / "encoded"))
    metadata = {
        "config": settings,
        "raw_rows": {name: len(data) for name, data in raw.items()},
        "split_rows": {name: len(data) for name, data in splits.items()},
        "tokenizer_training_rows": training_rows,
        "actual_vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": tokenizer_digest(output / "tokenizer.json"),
        "inputs": inputs,
        "validation_sha256": dataset_digest(encoded["val"]),
        "split_fingerprints": {
            name: data._fingerprint for name, data in encoded.items()
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared data: {output.resolve()}")
    print(metadata["split_rows"])
