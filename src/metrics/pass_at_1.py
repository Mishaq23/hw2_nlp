"""Single-program numeric agreement with executable validation references."""

import json
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from src.metrics.base_metric import BaseMetric


def execute_program(code, timeout_seconds=1.0, memory_mb=128):
    """Return a finite number, or None on invalid code, errors and timeouts."""
    try:
        process = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(Path(__file__).with_name("_execution_worker.py")),
            ],
            input=json.dumps(
                {
                    "code": code,
                    "timeout_seconds": timeout_seconds,
                    "memory_mb": memory_mb,
                }
            ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
            cwd=tempfile.gettempdir(),
            env={"PATH": os.defpath},
            check=True,
        )
        value = json.loads(process.stdout)
        return value if value is not None and math.isfinite(value) else None
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


class PassAt1(BaseMetric):
    def __init__(
        self,
        name="pass_at_1",
        rtol=1e-6,
        atol=1e-6,
        timeout_seconds=1.0,
        memory_mb=128,
        workers=2,
    ):
        super().__init__(name=name)
        self.rtol, self.atol = rtol, atol
        self.workers = workers
        self.execute = partial(
            execute_program, timeout_seconds=timeout_seconds, memory_mb=memory_mb
        )
        self.reference_results = {}

    @property
    def names(self):
        return (self.name, "reference_validity", "execution_validity")

    def _score(self, generated_code, reference_code):
        missing = list(
            dict.fromkeys(
                code for code in reference_code if code not in self.reference_results
            )
        )
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            self.reference_results.update(zip(missing, pool.map(self.execute, missing)))
            eligible = [
                index
                for index, code in enumerate(reference_code)
                if self.reference_results[code] is not None
            ]
            predictions = list(
                pool.map(self.execute, [generated_code[index] for index in eligible])
            )
        correct = numeric = 0
        for index, prediction in zip(eligible, predictions):
            if prediction is None:
                continue
            numeric += 1
            reference = self.reference_results[reference_code[index]]
            correct += abs(prediction - reference) <= self.atol + self.rtol * abs(
                reference
            )
        return correct, len(eligible), numeric

    def __call__(self, generated_code, reference_code, **batch):
        correct, eligible, _ = self._score(generated_code, reference_code)
        return correct / eligible if eligible else 0.0

    def update(self, tracker, generated_code, reference_code, **batch):
        correct, eligible, numeric = self._score(generated_code, reference_code)
        count = len(reference_code)
        tracker.update(self.name, correct / eligible if eligible else 0.0, n=eligible)
        tracker.update(
            "execution_validity", numeric / eligible if eligible else 0.0, n=eligible
        )
        tracker.update(
            "reference_validity", eligible / count if count else 0.0, n=count
        )
