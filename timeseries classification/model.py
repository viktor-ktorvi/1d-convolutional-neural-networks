from typing import List

import math

from torch import nn, Tensor
from torch.nn import functional as F

from utils.conv1d import Conv1dWithLength, configure_parameters


class TimeseriesClassifier(nn.Module):
    """
    A model made to classify timeseries.
    """

    def __init__(self, input_length: int,
                 num_classes: int,
                 hidden_channels: List[int],
                 kernel_sizes: List[int],
                 strides: List[int] = None,
                 dilations: List[int] = None,
                 paddings: List[int] = None,
                 input_channels: int = 1):
        super(TimeseriesClassifier, self).__init__()

        convolutions = []

        for i in range(len(hidden_channels)):
            if i == 0:
                # take input parameters
                in_length = input_length
                in_channels = input_channels
            else:
                # rake previous layer parameters
                in_length = convolutions[i - 1].output_length
                in_channels = hidden_channels[i - 1]

            out_channels = hidden_channels[i]
            kernel_size, stride, dilation, padding = configure_parameters(i, kernel_sizes, strides, dilations, paddings)

            convolutions.append(Conv1dWithLength(input_length=in_length,
                                                 in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 dilation=dilation))

        self.conv = nn.ModuleList(convolutions)

        self.linear = nn.Linear(in_features=convolutions[-1].output_length * hidden_channels[-1], out_features=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for i in range(len(self.conv)):
            out = self.conv[i](out)

            if i != len(self.conv) - 1:
                out = F.relu(out)
            else:
                out = F.tanh(out)

        out = out.reshape(x.shape[0], -1)
        return self.linear(out)
