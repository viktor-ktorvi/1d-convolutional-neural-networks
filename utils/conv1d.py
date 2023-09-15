import math
from typing import List, Tuple

from torch import nn


class Conv1dWithLength(nn.Conv1d):
    """
    Conv1d that keeps track of the input and output length
    """

    def __init__(self, input_length: int, *args, **kwargs):
        super(Conv1dWithLength, self).__init__(*args, **kwargs)

        self.input_length = input_length

        self.output_length = math.floor((self.input_length + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)


class ConvTranspose1dWithLength(nn.ConvTranspose1d):
    def __init__(self, input_length: int, *args, **kwargs):
        super(ConvTranspose1dWithLength, self).__init__(*args, **kwargs)

        self.input_length = input_length
        self.output_length = (input_length - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1


def configure_parameters(i: int, kernel_sizes: List[int], strides: List[int], dilations: List[int], paddings: List[int]) -> Tuple[int, ...]:
    """
    Extract the parameters of the i-th layer from their lists.
    :param i: Layer number.
    :param kernel_sizes:
    :param strides:
    :param dilations:
    :param paddings:
    :return:
    """
    kernel_size = kernel_sizes[i]

    if strides is None:
        stride = 1
    else:
        stride = strides[i]

    if dilations is None:
        dilation = 1
    else:
        dilation = dilations[i]

    if paddings is None:
        padding = math.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2)  # roughly keep the kernel from eating into the signal
        if padding < 0:
            padding = 0
    else:
        padding = paddings[i]

    return kernel_size, stride, dilation, padding
