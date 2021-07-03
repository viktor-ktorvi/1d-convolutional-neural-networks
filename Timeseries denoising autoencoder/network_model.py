import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sizes import *


class Network(nn.Module):
    def __init__(self, signal_len):
        super(Network, self).__init__()

        kernel_pool = 2
        kernel_conv1 = 5
        kernel_conv2 = 5
        out_conv1 = calcConv1dPoolOutSize(signal_len, kernel_conv1, kernel_pool)
        out_conv2 = calcConv1dPoolOutSize(out_conv1, kernel_conv2, kernel_pool)

        # TODO configure padding so that the edges look better
        self.pool = nn.MaxPool1d(kernel_pool)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=20, kernel_size=kernel_conv1)
        self.conv2 = nn.Conv1d(in_channels=20, out_channels=50, kernel_size=kernel_conv2)

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
