import torch.nn as nn
import torch.nn.functional as F

from utils.sizes import *


class Network(nn.Module):
    def __init__(self, signal_len):
        super(Network, self).__init__()

        conv1_sizes = Conv1dLayerSizes(in_ch=1, out_ch=20, kernel=5)
        pool1_kernel = 2

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=conv1_sizes.in_ch,
                      out_channels=conv1_sizes.out_ch,
                      kernel_size=conv1_sizes.kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool1_kernel)
        )

        conv2_sizes = Conv1dLayerSizes(in_ch=conv1_sizes.out_ch, out_ch=50, kernel=5)
        pool2_kernel = 2

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch,
                      out_channels=conv2_sizes.out_ch,
                      kernel_size=conv2_sizes.kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool2_kernel)
        )

        stride = 2
        out_deconv1 = round(signal_len / 2)
        # TODO foo for kernel size calc
        # Lout = (Lin - 1) * stride - 2*padding + dilation*(kernel_size - 1) + output_padding + 1
        kernel_deconv1 = out_deconv1 - stride * (out_conv2 - 1)
        kernel_deconv2 = signal_len - stride * (out_deconv1 - 1)

        self.deconv1 = nn.ConvTranspose1d(in_channels=50, out_channels=35, stride=stride, kernel_size=kernel_deconv1)
        self.deconv2 = nn.ConvTranspose1d(in_channels=35, out_channels=1, stride=stride, kernel_size=kernel_deconv2)

    def forward(self, x):
        # encoder
        # 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # decoder
        # 1
        x = self.deconv1(x)
        x = F.relu(x)

        # 2
        x = self.deconv2(x)

        return x
