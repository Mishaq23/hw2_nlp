"""Shuffle examples in pools, then form batches with similar sequence lengths."""

import numpy as np
from torch.utils.data import Sampler


class LengthGroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, pool_batches=64, seed=1):
        if batch_size < 1 or pool_batches < 1:
            raise ValueError("batch_size and pool_batches must be positive.")
        self.lengths = np.asarray(
            dataset.data["source_length"], dtype=np.int32
        ) + np.asarray(dataset.data["target_length"], dtype=np.int32)
        self.batch_size = batch_size
        self.pool_size = batch_size * pool_batches
        self.seed = seed
        self.iteration = 0

    def __len__(self):
        return len(self.lengths) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.iteration)
        self.iteration += 1
        indices = rng.permutation(len(self.lengths))
        for start in range(0, len(indices), self.pool_size):
            pool = indices[start : start + self.pool_size]
            pool = pool[np.argsort(self.lengths[pool], kind="stable")]
            full_size = len(pool) // self.batch_size * self.batch_size
            batches = pool[:full_size].reshape(-1, self.batch_size)
            for index in rng.permutation(len(batches)):
                yield batches[index].tolist()
