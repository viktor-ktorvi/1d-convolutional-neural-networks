import math

from torch import nn


class Conv1dWithLength(nn.Conv1d):
    """
    Conv1d that keeps track of the input and output length
    """

    def __init__(self, input_length: int, *args, **kwargs):
        super(Conv1dWithLength, self).__init__(*args, **kwargs)

        self.input_length = input_length

        self.output_length = math.floor((self.input_length + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
