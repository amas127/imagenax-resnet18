import typing as tp
from collections.abc import Sequence
from pathlib import Path

import grain
import jax
import jax.numpy as jnp
import numpy as np

from .transforms import (
    EvalCrop,
    RandomFlip,
    RandomResizedCrop,
    ReadRecord,
)


def batch_fn(records: Sequence[dict[str, tp.Any]]):
    images = []
    labels = []
    for record in records:
        images.append(record["image"])
        labels.append(record["label"])
    images = np.stack(images, axis=0)
    labels = np.stack(labels, axis=0)
    return images, labels


def get_train_dataloader(
    pattern: str,
    nepochs: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    worker_count: int = 0,
):
    paths = sorted(Path.cwd().glob(pattern))
    if not len(paths):
        raise LookupError
    source = grain.sources.ArrayRecordDataSource(paths)
    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        num_epochs=nepochs,
        shard_options=grain.sharding.NoSharding(),
        shuffle=shuffle,
        seed=seed,
    )
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[
            ReadRecord(),
            RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            RandomFlip(),
            grain.transforms.Batch(batch_size, True, batch_fn),
        ],
        worker_count=worker_count,
        worker_buffer_size=1,
        read_options=grain.ReadOptions(),
    )


def get_eval_dataloader(
    pattern: str,
    batch_size: int,
    worker_count: int = 0,
):
    paths = sorted(Path.cwd().glob(pattern))
    if not len(paths):
        raise LookupError
    source = grain.sources.ArrayRecordDataSource(paths)
    sampler = grain.samplers.SequentialSampler(
        num_records=len(source),
        shard_options=grain.sharding.NoSharding(),
    )
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[
            ReadRecord(),
            EvalCrop(),
            grain.transforms.Batch(batch_size, False, batch_fn),
        ],
        worker_count=worker_count,
        worker_buffer_size=1,
        read_options=grain.ReadOptions(),
    )


@jax.jit
def normalise(samples: jax.Array):
    mean = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
    std = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
    return (samples.astype(jnp.float32) / 255.0 - mean) / std
