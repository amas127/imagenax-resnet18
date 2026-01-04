from collections.abc import Sequence

import flax.typing as ftp
import jax.numpy as jnp
from flax import nnx

from .block import BasicBlock, ResNetBlock, Stem
from .constants import FC_BIAS_INIT, FC_WEIGHT_INIT, STEM_OUT_FEATURES, OutputType


class ResNet(nnx.Module):
    def __init__(
        self,
        in_features: int,
        block: type[ResNetBlock],
        stage_sizes: Sequence[int],
        num_classes: int,
        output_type: OutputType = "logits",
        *,
        dtype: ftp.Dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.output_type = output_type
        self.stem = Stem(in_features, dtype=dtype, rngs=rngs)

        current_channels = STEM_OUT_FEATURES
        self.stages = nnx.List([])

        for i, num_blocks in enumerate(stage_sizes):
            is_first_stage = i == 0
            out_channels = current_channels if is_first_stage else current_channels * 2

            blocks = []
            for j in range(num_blocks):
                should_downsample = (j == 0) and not is_first_stage

                blocks.append(
                    block(
                        din=current_channels,
                        dout=out_channels,
                        downsample=should_downsample,
                        dtype=dtype,
                        rngs=rngs,
                    )
                )
                current_channels = out_channels

            self.stages.append(nnx.Sequential(*blocks))

        self.linear = nnx.Linear(
            current_channels,
            num_classes,
            use_bias=True,
            kernel_init=FC_WEIGHT_INIT,
            bias_init=FC_BIAS_INIT,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: ftp.Array) -> ftp.Array | dict[str, ftp.Array]:
        activations = {}
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            activations[f"stage{i}"] = x

        x = jnp.mean(x, axis=(1, 2))
        logits = self.linear(x)

        if self.output_type == "activations":
            activations["logits"] = logits
            return activations

        return {
            "logits": logits,
            "softmax": nnx.softmax(logits, axis=-1),
            "log_softmax": nnx.log_softmax(logits, axis=-1),
        }[self.output_type]


def get_resnet18(
    in_features: int = 3,
    num_classes: int = 1000,
    output_type: OutputType = "logits",
    *,
    dtype: ftp.Dtype = jnp.bfloat16,
    rngs: nnx.Rngs,
):
    return ResNet(
        in_features,
        BasicBlock,
        [2, 2, 2, 2],
        num_classes,
        output_type,
        dtype=dtype,
        rngs=rngs,
    )


def get_resnet34(
    in_features: int = 3,
    num_classes: int = 1000,
    output_type: OutputType = "logits",
    *,
    dtype: ftp.Dtype = jnp.bfloat16,
    rngs: nnx.Rngs,
):
    return ResNet(
        in_features,
        BasicBlock,
        [3, 4, 6, 3],
        num_classes,
        output_type,
        dtype=dtype,
        rngs=rngs,
    )
