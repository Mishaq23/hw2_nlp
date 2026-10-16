"""Replayable batches whose cursor advances only after the trainer consumes them."""

import torch
from torch.utils.data import Sampler


class ResumableBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, seed=0):
        self.size = len(dataset)
        self.batch_size = batch_size
        self.seed = seed
        self.fingerprint = getattr(getattr(dataset, "data", None), "_fingerprint", None)
        self.pass_index = 0
        self.batch_offset = 0

    def __len__(self):
        # Include the final short batch so a full pass covers every example.
        return (self.size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.pass_index)
        order = torch.randperm(self.size, generator=generator)
        for index in range(self.batch_offset, len(self)):
            start = index * self.batch_size
            yield order[start : start + self.batch_size].tolist()

    def mark_consumed(self):
        # DataLoader workers may prefetch ahead of this cursor.
        self.batch_offset += 1
        if self.batch_offset == len(self):
            self.pass_index += 1
            self.batch_offset = 0

    def state_dict(self):
        return {
            "size": self.size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "fingerprint": self.fingerprint,
            "pass_index": self.pass_index,
            "batch_offset": self.batch_offset,
        }

    def load_state_dict(self, state):
        for name in ("size", "batch_size", "seed", "fingerprint"):
            if state[name] != getattr(self, name):
                raise ValueError(
                    f"Training data/batching changed ({name}). Set "
                    "trainer.reset_data_state=true to start a new data group."
                )
        self.pass_index = state["pass_index"]
        self.batch_offset = state["batch_offset"]
