import typing as tp

from flax import nnx

OutputType: tp.TypeAlias = tp.Literal["softmax", "log_softmax", "activations", "logits"]

CONV_KERNEL_INIT = nnx.initializers.he_normal()
BN_EPSILON = 1e-5
BN_MOMENTUM = 0.9
BN_SCALE_INIT = nnx.initializers.ones_init()
BN_BIAS_INIT = nnx.initializers.zeros_init()
STEM_CONV_KERNEL_SIZE = (7, 7)
DOWNSAMPLE_STRIDES = (2, 2)
STEM_MAXPOOL_KERNEL_SIZE = (3, 3)
PADDING = "SAME_LOWER"
CONV_KERNEL_SIZE = (3, 3)
CONV_STRIDES = (1, 1)
RESCONV_KERNEL_SIZE = (1, 1)
STEM_OUT_FEATURES = 64
FC_WEIGHT_INIT = nnx.initializers.he_normal()
FC_BIAS_INIT = nnx.initializers.zeros_init()
