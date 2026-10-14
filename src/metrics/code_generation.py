"""Token metrics for teacher forcing and syntax checks for generated code."""

import math

import torch
from torch.nn import functional as F

from src.metrics.base_metric import BaseMetric


class _TokenMetric(BaseMetric):
    def __init__(self, ignore_index=-100, name=None):
        super().__init__(name=name)
        self.ignore_index = ignore_index

    def weight(self, labels, **batch):
        return labels.ne(self.ignore_index).sum().item()


class TokenAccuracy(_TokenMetric):
    @torch.no_grad()
    def __call__(self, logits, labels, **batch):
        valid = labels.ne(self.ignore_index)
        correct = (logits.argmax(-1).eq(labels) & valid).sum().item()
        count = self.weight(labels)
        return correct / count if count else 0.0


class Perplexity(_TokenMetric):
    @torch.no_grad()
    def _mean_nll(self, logits, labels):
        # Plain NLL, independent of label smoothing in the training loss.
        return F.cross_entropy(
            logits.float().flatten(0, 1),
            labels.flatten(),
            ignore_index=self.ignore_index,
        ).item()

    def __call__(self, logits, labels, **batch):
        return self.compute(self._mean_nll(logits, labels))

    def update(self, tracker, logits, labels, **batch):
        # Average NLL across tokens first; averaging batch perplexities is wrong.
        tracker.update(self.name, self._mean_nll(logits, labels), n=self.weight(labels))

    def compute(self, average):
        try:
            return math.exp(average)
        except OverflowError:
            return float("inf")


class SyntaxValidity(BaseMetric):
    def weight(self, generated_code, **batch):
        return len(generated_code)

    def __call__(self, generated_code, **batch):
        valid = 0
        for code in generated_code:
            if not code.strip():
                continue
            try:
                # Compile checks syntax and contexts such as return/break,
                # without executing the generated program.
                compile(code, "<generated>", "exec")
                valid += 1
            except (SyntaxError, ValueError):
                pass
        return valid / len(generated_code) if generated_code else 0.0
