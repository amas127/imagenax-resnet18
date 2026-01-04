import jax
import optax
from flax import nnx

from .data import normalise


def get_optimizer_and_scheduler(model: nnx.Module):
    lr_init = 1e-1
    scales = {int(15e4): 1e-1, int(30e4): 1e-1, int(45e4): 1e-1}
    momentum = 0.9
    nesterov = False
    weight_decay = 1e-4

    def map_fn(path, arr):
        path = "/".join(str(p) for p in path).lower()
        return not any(sub in path for sub in ("bn", "bias"))

    wd = optax.masked(
        optax.add_decayed_weights(weight_decay),
        lambda p: jax.tree.map_with_path(map_fn, p),
    )
    scheduler = optax.schedules.piecewise_constant_schedule(lr_init, scales)
    sgd = optax.sgd(scheduler, momentum=momentum, nesterov=nesterov)
    return nnx.Optimizer(model, optax.chain(wd, sgd), wrt=nnx.Param), scheduler


def compute_loss(model: nnx.Module, samples: jax.Array, labels: jax.Array):
    out = model(normalise(samples))
    logits = out["logits"] if isinstance(out, dict) else out
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss, logits


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    tracer: nnx.MultiMetric,
    samples: jax.Array,
    labels: jax.Array,
):
    grad_fn = nnx.value_and_grad(compute_loss, has_aux=True)
    (loss, logits), grads = grad_fn(model, samples, labels)
    tracer.update(loss=loss, logits=logits, labels=labels)
    optimizer.update(model, grads)


@nnx.jit
def eval_step(
    model: nnx.Module,
    tracer: nnx.MultiMetric,
    samples: jax.Array,
    labels: jax.Array,
):
    loss, logits = compute_loss(model, samples, labels)
    tracer.update(loss=loss, logits=logits, labels=labels)
