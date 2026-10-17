"""Sample every source Parquet file without caching the full corpus locally."""

from pathlib import Path

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


def sample_parquet_files(dataset, output_path, max_rows, seed=42, batch_size=4096):
    """Write a proportional, reproducible sample; return files and source counts.

    Only metadata is read in the first pass. The second pass reads one file at
    a time, keeping selected row indices and a batch rather than the corpus.
    The row limit applies before missing-text filtering and train/val splitting.
    """
    if max_rows < 1 or batch_size < 1:
        raise ValueError("sampling.max_rows and sampling.batch_size must be positive.")
    data_files = dataset.get("data_files")
    if not isinstance(data_files, dict) or not {"train", "test"} <= data_files.keys():
        raise ValueError("Sampling requires dataset.data_files with train and test.")

    def source_paths(split):
        paths = data_files[split]
        if isinstance(paths, str):
            paths = [paths]
        if dataset["path"] != "parquet":
            revision = dataset.get("revision") or "main"
            prefix = f"hf://datasets/{dataset['path']}@{revision}/"
            paths = [
                path if "://" in path or Path(path).is_absolute() else prefix + path
                for path in paths
            ]
        return paths

    fs, _, files = fsspec.get_fs_token_paths(source_paths("train"), mode="rb")
    files = sorted(set(files))
    if not files:
        raise ValueError("No training Parquet files match dataset.data_files.train.")
    counts = []
    for path in tqdm(files, desc="Read source shard sizes"):
        # The file handle has an in-memory buffer only; no full-file disk cache.
        with fs.open(path, "rb", cache_type="none") as stream:
            counts.append(pq.ParquetFile(stream).metadata.num_rows)
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    limit = min(max_rows, total)
    if limit < np.count_nonzero(counts) or not total:
        raise ValueError("sampling.max_rows must cover every nonempty source shard.")

    # Largest-remainder allocation preserves the total and respects shard sizes.
    quotas, remainders = np.divmod(limit * counts, total)
    remaining = limit - int(quotas.sum())
    quotas[np.argsort(-remainders, kind="stable")[:remaining]] += 1
    for index in np.flatnonzero((counts > 0) & (quotas == 0)):
        donor = int(quotas.argmax())
        quotas[donor] -= 1
        quotas[index] = 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("question", pa.string()), ("code", pa.string())])
    rng = np.random.default_rng(seed)
    shards = []
    with pq.ParquetWriter(output_path, schema) as writer:
        for path, size, quota in tqdm(
            zip(files, counts, quotas), total=len(files), desc="Sample source shards"
        ):
            indices = np.sort(rng.choice(size, size=quota, replace=False))
            start = 0
            if quota:
                with fs.open(path, "rb", cache_type="none") as stream:
                    parquet = pq.ParquetFile(stream, pre_buffer=False)
                    for batch in parquet.iter_batches(
                        batch_size=batch_size, columns=schema.names
                    ):
                        end = start + batch.num_rows
                        left, right = np.searchsorted(indices, [start, end])
                        if right > left:
                            selected = batch.take(indices[left:right] - start)
                            writer.write_table(
                                pa.Table.from_batches([selected]).cast(schema)
                            )
                        start = end
                        if start > indices[-1]:
                            break
            shards.append(
                {
                    "file": fs.unstrip_protocol(path),
                    "source_rows": int(size),
                    "sampled_rows": int(quota),
                }
            )
    metadata = {"source_rows": total, "sampled_rows": limit, "shards": shards}
    return {"train": str(output_path), "test": source_paths("test")}, metadata
