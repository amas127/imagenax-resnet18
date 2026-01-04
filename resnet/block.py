from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import partial

import flax.typing as ftp
from flax import nnx

from .constants import (
    BN_BIAS_INIT,
    BN_EPSILON,
    BN_MOMENTUM,
    BN_SCALE_INIT,
    CONV_KERNEL_INIT,
    CONV_KERNEL_SIZE,
    CONV_STRIDES,
    DOWNSAMPLE_STRIDES,
    PADDING,
    RESCONV_KERNEL_SIZE,
    STEM_CONV_KERNEL_SIZE,
    STEM_MAXPOOL_KERNEL_SIZE,
    STEM_OUT_FEATURES,
)


class ResNetBlock(nnx.Module, ABC):
    @abstractmethod
    def __init__(
        self,
        din: int,
        dout: int,
        downsample: bool,
        *,
        dtype: ftp.Dtype,
        rngs: nnx.Rngs,
    ): ...


def get_conv(
    din: int,
    dout: int,
    kernel_size: Sequence[int],
    strides: Sequence[int],
    *,
    dtype: ftp.Dtype | None,
    rngs: nnx.Rngs,
):
    return nnx.Conv(
        din,
        dout,
        kernel_size,
        strides,
        padding=PADDING,
        use_bias=False,
        kernel_init=CONV_KERNEL_INIT,
        dtype=dtype,
        rngs=rngs,
    )


def get_bn(
    dim: int,
    *,
    dtype: ftp.Dtype | None,
    rngs: nnx.Rngs,
):
    return nnx.BatchNorm(
        dim,
        momentum=BN_MOMENTUM,
        epsilon=BN_EPSILON,  # epsilon: 1e-5 (standard) -> 1e-3
        use_scale=True,
        use_bias=True,
        scale_init=BN_SCALE_INIT,
        bias_init=BN_BIAS_INIT,
        dtype=dtype,
        rngs=rngs,
    )


class Stem(nnx.Module):
    def __init__(
        self,
        din: int,
        *,
        dtype: ftp.Dtype,
        rngs: nnx.Rngs,
    ):
        self.conv = get_conv(
            din,
            STEM_OUT_FEATURES,
            STEM_CONV_KERNEL_SIZE,
            DOWNSAMPLE_STRIDES,
            dtype=dtype,
            rngs=rngs,
        )
        self.bn = get_bn(STEM_OUT_FEATURES, dtype=dtype, rngs=rngs)
        self.relu = nnx.relu
        self.mp = partial(
            nnx.max_pool,
            window_shape=STEM_MAXPOOL_KERNEL_SIZE,
            strides=DOWNSAMPLE_STRIDES,
            padding=PADDING,
        )

    def __call__(self, x: ftp.Array) -> ftp.Array:
        return self.mp(self.relu(self.bn(self.conv(x))))


class BasicBlock(ResNetBlock):
    def __init__(
        self,
        din: int,
        dout: int,
        downsample: bool,
        *,
        dtype: ftp.Dtype,
        rngs: nnx.Rngs,
    ):
        self.conv1 = get_conv(
            din,
            dout,
            CONV_KERNEL_SIZE,
            DOWNSAMPLE_STRIDES if downsample else CONV_STRIDES,
            dtype=dtype,
            rngs=rngs,
        )
        self.bn1 = get_bn(dout, dtype=dtype, rngs=rngs)
        self.relu1 = nnx.relu
        self.conv2 = get_conv(
            dout,
            dout,
            CONV_KERNEL_SIZE,
            CONV_STRIDES,
            dtype=dtype,
            rngs=rngs,
        )
        self.bn2 = get_bn(dout, dtype=dtype, rngs=rngs)
        self.relu2 = nnx.relu

        if downsample:
            self.res_conv = get_conv(
                din,
                dout,
                RESCONV_KERNEL_SIZE,
                DOWNSAMPLE_STRIDES,
                dtype=dtype,
                rngs=rngs,
            )
            self.res_bn = get_bn(dout, dtype=dtype, rngs=rngs)
        else:
            self.res_conv = nnx.identity
            self.res_bn = nnx.identity

    def __call__(self, x: ftp.Array) -> ftp.Array:
        residual = self.res_bn(self.res_conv(x))
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu2(x + residual)
