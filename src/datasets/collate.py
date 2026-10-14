from functools import partial
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from src.datasets.tokenizer import CodeTokenizer


def code_generation_collate_fn(dataset_items: list[dict], pad_id: int) -> dict:
    """Right-pad seq2seq inputs on CPU; True in a mask means a real token.

    Dataset already shifts decoder inputs and labels. Label padding is -100
    for cross-entropy's ignore_index; test batches only contain encoder inputs.
    """
    if not dataset_items:
        raise ValueError("Cannot collate an empty batch.")
    has_targets = "labels" in dataset_items[0]
    for item in dataset_items:
        if ("labels" in item) != has_targets or (
            "decoder_input_ids" in item
        ) != has_targets:
            raise ValueError(
                "Every example must consistently contain both target fields."
            )
        if has_targets and len(item["decoder_input_ids"]) != len(item["labels"]):
            raise ValueError("Decoder inputs and labels must have the same length.")

    encoder_ids, encoder_mask = _pad_inputs(
        [item["encoder_input_ids"] for item in dataset_items], pad_id
    )
    batch = {
        "encoder_input_ids": encoder_ids,
        "encoder_attention_mask": encoder_mask,
        "sample_id": [item["sample_id"] for item in dataset_items],
    }
    if has_targets:
        decoder_ids, decoder_mask = _pad_inputs(
            [item["decoder_input_ids"] for item in dataset_items], pad_id
        )
        batch.update(
            decoder_input_ids=decoder_ids,
            decoder_attention_mask=decoder_mask,
            labels=pad_sequence(
                [
                    torch.as_tensor(item["labels"], dtype=torch.long, device="cpu")
                    for item in dataset_items
                ],
                batch_first=True,
                padding_value=-100,
            ),
        )
    return batch


def _pad_inputs(sequences, pad_id):
    tensors = [
        torch.as_tensor(sequence, dtype=torch.long, device="cpu")
        for sequence in sequences
    ]
    padded = pad_sequence(tensors, batch_first=True, padding_value=pad_id)
    lengths = torch.tensor([len(sequence) for sequence in tensors])
    mask = torch.arange(padded.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, mask


def make_code_generation_collate(tokenizer_path: str | Path):
    """Bind the saved tokenizer's PAD ID; the resulting partial is picklable."""
    pad_id = CodeTokenizer.from_file(tokenizer_path).pad_id
    return partial(code_generation_collate_fn, pad_id=pad_id)
