import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax


class TopkAccuracy(nnx.metrics.Average):
    @tp.override
    def __init__(self, k: int):
        self.k = k
        super().__init__()

    @tp.override
    def update(self, *, logits: jax.Array, labels: jax.Array, **_) -> None:
        labels = jnp.astype(labels, jnp.int32)
        _, topk_idx = lax.top_k(logits, self.k, axis=-1)
        super().update(values=jnp.any(topk_idx == labels[:, None], axis=-1))


def get_tracer():
    return nnx.metrics.MultiMetric(
        top1acc=TopkAccuracy(1),
        top5acc=TopkAccuracy(5),
        loss=nnx.metrics.Average("loss"),
    )
