import jax
from flax import nnx

from resnet.model import get_resnet18


def count_params():
    rngs = nnx.Rngs(0)
    model = get_resnet18(rngs=rngs)

    # Get all parameters
    params = nnx.state(model, nnx.Param)

    # Count total elements
    total_params = sum(x.size for x in jax.tree.leaves(params))

    print(f"Total parameters: {total_params:,}")
    return total_params


if __name__ == "__main__":
    count_params()
