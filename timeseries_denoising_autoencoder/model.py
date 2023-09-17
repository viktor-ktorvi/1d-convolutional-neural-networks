from typing import List

from torch import nn, Tensor
from torch.nn import functional as F

from utils.conv1d import Conv1dWithLength, configure_parameters, ConvTranspose1dWithLength


class TimeseriesAutoencoder(nn.Module):
    """
    A model made to classify timeseries.
    """

    def __init__(self, input_length: int,
                 hidden_channels: List[int],
                 kernel_sizes: List[int],
                 strides: List[int] = None,
                 dilations: List[int] = None,
                 paddings: List[int] = None,
                 input_channels: int = 1):
        super(TimeseriesAutoencoder, self).__init__()

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

        transpose_convolutions = []
        for i in range(len(convolutions) - 1, -1, -1):
            transpose_convolutions.append(ConvTranspose1dWithLength(
                input_length=convolutions[i].output_length,
                in_channels=convolutions[i].out_channels,
                out_channels=convolutions[i].in_channels,
                kernel_size=convolutions[i].kernel_size,
                stride=convolutions[i].stride,
                padding=convolutions[i].padding,
                output_padding=1 if convolutions[i].stride[0] > 1 else 0,
                # padding_mode="reflect"
            ))

        self.conv = nn.ModuleList(convolutions)
        self.transpose_conv = nn.ModuleList(transpose_convolutions)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for i in range(len(self.conv)):
            out = self.conv[i](out)

            if i != len(self.conv) - 1:
                out = F.relu(out)
            else:
                out = F.tanh(out)

        for i in range(len(self.transpose_conv)):
            out = self.transpose_conv[i](out)

            if i != len(self.transpose_conv) - 1:
                out = F.relu(out)

        return out.squeeze()
