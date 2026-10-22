"""Shuffle examples in pools, then form batches with similar sequence lengths."""

import numpy as np
from torch.utils.data import Sampler


class LengthGroupedBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        pool_batches=64,
        seed=1,
        max_tokens=None,
        pad_to_multiple_of=1,
    ):
        if batch_size < 1 or pool_batches < 1:
            raise ValueError("batch_size and pool_batches must be positive.")
        if (max_tokens is not None and max_tokens < 1) or pad_to_multiple_of < 1:
            raise ValueError("max_tokens and pad_to_multiple_of must be positive.")
        self.source_lengths = np.asarray(dataset.data["source_length"], dtype=np.int32)
        self.target_lengths = np.asarray(dataset.data["target_length"], dtype=np.int32)
        self.lengths = self.source_lengths + self.target_lengths
        self.batch_size = batch_size
        self.pool_size = batch_size * pool_batches
        self.seed = seed
        self.iteration = 0
        self.max_tokens = max_tokens
        self.pad_to_multiple_of = pad_to_multiple_of

    def __len__(self):
        if self.max_tokens is not None:
            return sum(1 for _ in self._batches(self.seed + self.iteration))
        return len(self.lengths) // self.batch_size

    def __iter__(self):
        seed = self.seed + self.iteration
        self.iteration += 1
        yield from self._batches(seed)

    def _batches(self, seed):
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self.lengths))
        for start in range(0, len(indices), self.pool_size):
            pool = indices[start : start + self.pool_size]
            pool = pool[np.argsort(self.lengths[pool], kind="stable")]
            if self.max_tokens is None:
                full_size = len(pool) // self.batch_size * self.batch_size
                batches = pool[:full_size].reshape(-1, self.batch_size).tolist()
            else:
                batches = self._pack_pool(pool)
            for index in rng.permutation(len(batches)):
                yield batches[index]

    def _pack_pool(self, pool):
        """Keep complete examples; a single oversized example gets its own batch."""
        batches, batch = [], []
        source_max = target_max = 0
        multiple = self.pad_to_multiple_of
        for index in pool:
            source = (
                (int(self.source_lengths[index]) + multiple - 1) // multiple * multiple
            )
            target = (
                (int(self.target_lengths[index]) + multiple - 1) // multiple * multiple
            )
            next_source, next_target = max(source_max, source), max(target_max, target)
            if batch and (
                len(batch) == self.batch_size
                or (len(batch) + 1) * (next_source + next_target) > self.max_tokens
            ):
                batches.append(batch)
                batch = []
                source_max = target_max = 0
            batch.append(int(index))
            source_max, target_max = max(source_max, source), max(target_max, target)
        if batch:
            batches.append(batch)
        return batches
