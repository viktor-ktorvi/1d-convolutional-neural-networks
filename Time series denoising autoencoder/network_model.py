import torch.nn as nn

from utils.sizes import Conv1dLayerSizes, TransposeConv1dLayerSizes


class Network(nn.Module):
    def __init__(self, signal_len):
        super(Network, self).__init__()

        # encoder
        conv1_sizes = Conv1dLayerSizes(in_len=signal_len,
                                       in_ch=1,
                                       out_ch=20,
                                       kernel=5,
                                       pool_kernel=2)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=conv1_sizes.in_ch,
                      out_channels=conv1_sizes.out_ch,
                      kernel_size=conv1_sizes.kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=conv1_sizes.pool_kernel)
        )

        conv2_sizes = Conv1dLayerSizes(in_len=conv1_sizes.out_len,
                                       in_ch=conv1_sizes.out_ch,
                                       out_ch=50,
                                       kernel=5,
                                       pool_kernel=2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch,
                      out_channels=conv2_sizes.out_ch,
                      kernel_size=conv2_sizes.kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=conv2_sizes.pool_kernel)
        )

        # decoder

        deconv1_sizes = TransposeConv1dLayerSizes(in_len=conv2_sizes.out_len,
                                                  out_len=round(signal_len / 2),
                                                  in_ch=conv2_sizes.out_ch,
                                                  out_ch=35,
                                                  stride=2)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=deconv1_sizes.in_ch,
                               out_channels=deconv1_sizes.out_ch,
                               stride=deconv1_sizes.stride,
                               kernel_size=deconv1_sizes.kernel_size),
            nn.ReLU()
        )

        deconv2_sizes = TransposeConv1dLayerSizes(in_len=deconv1_sizes.out_len,
                                                  out_len=signal_len,
                                                  in_ch=deconv1_sizes.out_ch,
                                                  out_ch=1,
                                                  stride=2)

        self.deconv2 = nn.ConvTranspose1d(in_channels=deconv2_sizes.in_ch,
                                          out_channels=deconv2_sizes.out_ch,
                                          stride=deconv2_sizes.stride,
                                          kernel_size=deconv2_sizes.kernel_size)

    def forward(self, x):
        # encoder
        x = self.conv1(x)
        x = self.conv2(x)

        # decoder
        x = self.deconv1(x)
        x = self.deconv2(x)

        return x

