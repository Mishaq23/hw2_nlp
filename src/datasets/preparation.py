"""Prepare TinySMS once and reuse the saved tokenizer and Arrow splits."""

import json
from pathlib import Path

from omegaconf import OmegaConf

from datasets import DatasetDict, load_dataset
from src.datasets.code_generation import tokenize_dataset
from src.datasets.tinysms import prepare_raw_splits
from src.datasets.tokenizer import CodeTokenizer


def prepare_data(config):
    settings = OmegaConf.to_container(config, resolve=True)
    if not settings.pop("enabled", True):
        return
    output = Path(settings["output_dir"])
    metadata_path = output / "metadata.json"
    cached_files = [output / "tokenizer.json", metadata_path] + [
        output / "encoded" / split / "state.json" for split in ("train", "val", "test")
    ]
    if all(path.is_file() for path in cached_files):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["config"] == settings:
            print(f"Using prepared data: {output.resolve()}")
            return

    raw = load_dataset(**settings["dataset"])
    splits = prepare_raw_splits(raw, **settings["split"])
    tokenizer_train = splits["train"]
    limit = settings["tokenizer_train_rows"]
    if limit is not None:
        if limit < 1:
            raise ValueError("tokenizer_train_rows must be positive or null.")
        tokenizer_train = tokenizer_train.shuffle(
            seed=settings["split"]["seed"]
        ).select(range(min(limit, len(tokenizer_train))))
    tokenizer = CodeTokenizer.train(tokenizer_train, **settings["tokenizer"])
    encoded = DatasetDict(
        {
            name: tokenize_dataset(
                data, tokenizer, has_targets=name != "test", **settings["tokenize"]
            )
            for name, data in splits.items()
        }
    )
    tokenizer.save(output / "tokenizer.json")
    encoded.save_to_disk(str(output / "encoded"))
    metadata = {
        "config": settings,
        "raw_rows": {name: len(data) for name, data in raw.items()},
        "split_rows": {name: len(data) for name, data in splits.items()},
        "tokenizer_training_rows": len(tokenizer_train),
        "actual_vocab_size": tokenizer.vocab_size,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared data: {output.resolve()}")
    print(metadata["split_rows"])
