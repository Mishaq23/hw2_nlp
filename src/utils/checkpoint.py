"""Portable training state, independent of a particular model architecture."""

import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from src.datasets.code_generation import dataset_digest
from src.datasets.tokenizer import tokenizer_digest


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        for device, device_state in enumerate(
            state["cuda"][: torch.cuda.device_count()]
        ):
            torch.cuda.set_rng_state(device_state, device)


def dataset_identity(dataset):
    data = getattr(dataset, "data", None)
    return {
        "rows": len(dataset),
        "fingerprint": (
            dataset_digest(data) if hasattr(data, "column_names") else None
        ),
    }


def atomic_save(state, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_resume_checkpoint(config, root):
    """Read once on CPU and recover architecture/tokenizer before preparation."""
    resume = config.trainer.get("resume_from")
    if resume is None:
        return None
    path = Path(resume)
    if not path.is_absolute():
        path = root / config.trainer.save_dir / config.writer.run_name / path
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    next_epoch = checkpoint["epoch"] + int(checkpoint.get("epoch_complete", True))
    if config.trainer.n_epochs < next_epoch:
        raise ValueError(
            "trainer.n_epochs is the total budget across sessions; increase it "
            "beyond the saved epoch to continue training."
        )
    saved = checkpoint["config"]
    saved = saved if OmegaConf.is_config(saved) else OmegaConf.create(saved)
    if (
        checkpoint.get("tokenizer_sha256") is None
        and (path.parent / "tokenizer.json").is_file()
    ):
        checkpoint["tokenizer_sha256"] = tokenizer_digest(
            path.parent / "tokenizer.json"
        )
    with open_dict(config):
        config.trainer.resume_from = str(path.resolve())
        tokenizer_path = config.model.get("tokenizer_path")
        if config.trainer.get("resume_model_config", True):
            config.model = OmegaConf.create(
                OmegaConf.to_container(saved.model, resolve=False)
            )
            if "tokenizer_path" in config.model:
                preparation = config.get("data_preparation")
                if preparation is not None and preparation.get("enabled", True):
                    tokenizer_path = str(
                        Path(preparation.output_dir) / "tokenizer.json"
                    )
                elif tokenizer_path is None or not Path(tokenizer_path).is_file():
                    tokenizer_path = str(path.parent / "tokenizer.json")
                config.model.tokenizer_path = tokenizer_path
        if (
            config.writer.run_name == saved.writer.run_name
            and config.writer._target_ == saved.writer._target_
        ):
            config.writer.run_id = saved.writer.get("run_id")
        preparation = config.get("data_preparation")
        if preparation is not None and preparation.get("enabled", True):
            if (
                preparation.get("pretrained_tokenizer_path") is None
                or not Path(preparation.pretrained_tokenizer_path).is_file()
            ):
                preparation.pretrained_tokenizer_path = str(
                    path.parent / "tokenizer.json"
                )
            if (
                checkpoint.get("tokenizer_sha256") is not None
                and tokenizer_digest(preparation.pretrained_tokenizer_path)
                != checkpoint["tokenizer_sha256"]
            ):
                raise ValueError("Checkpoint and pretrained tokenizer do not match.")
            if (
                preparation.get("validation_path") is None
                or not Path(preparation.validation_path).is_dir()
            ):
                validation = path.parent / "validation"
                if not validation.is_dir():
                    # Older runs kept validation only in the prepared-data folder.
                    validation = Path(saved.datasets.val.data)
                if validation.is_dir():
                    preparation.validation_path = str(validation)
                else:
                    warnings.warn(
                        "No saved validation found for this legacy checkpoint. "
                        "Provide data_preparation.validation_path to keep its validation; "
                        "otherwise this stage will create a new one.",
                        RuntimeWarning,
                    )
        if (
            "T_max" in config.lr_scheduler
            and config.lr_scheduler.T_max is None
            and not config.trainer.get("reset_scheduler", False)
        ):
            config.lr_scheduler.T_max = saved.lr_scheduler.T_max
    return checkpoint
