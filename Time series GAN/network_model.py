import torch.nn as nn
from utils.sizes import Conv1dLayerSizes, TransposeConv1dLayerSizes, FullyConnectedLayerSizes


class Generator(nn.Module):
    def __init__(self, signal_len, noise_len):
        super(Generator, self).__init__()

        self.noise_len = noise_len

        deconv1_sizes = TransposeConv1dLayerSizes(in_len=self.noise_len,
                                                  out_len=round(signal_len / 2),
                                                  in_ch=1,
                                                  out_ch=20,
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
        x = self.deconv1(x)
        x = self.deconv2(x)

        return x


class Discriminator(nn.Module):

    def __init__(self, signal_len):
        super(Discriminator, self).__init__()

        conv1_sizes = Conv1dLayerSizes(in_len=signal_len,
                                       in_ch=1,
                                       out_ch=6,
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
                                       out_ch=15,
                                       kernel=5,
                                       pool_kernel=2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_sizes.in_ch,
                      out_channels=conv2_sizes.out_ch,
                      kernel_size=conv2_sizes.kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=conv2_sizes.pool_kernel)
        )

        # self. because we need it in self.forward
        self.fc1_sizes = FullyConnectedLayerSizes(in_features=conv2_sizes.out_len * conv2_sizes.out_ch,
                                                  out_features=round(0.7 * conv2_sizes.out_len * conv2_sizes.out_ch))

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.fc1_sizes.in_features, out_features=self.fc1_sizes.out_features),
            nn.ReLU()
        )

        fc2_sizes = FullyConnectedLayerSizes(in_features=self.fc1_sizes.out_features,
                                             out_features=round(0.5 * self.fc1_sizes.out_features))

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=fc2_sizes.in_features, out_features=fc2_sizes.out_features),
            nn.ReLU()
        )

        fc3_sizes = FullyConnectedLayerSizes(in_features=fc2_sizes.out_features,
                                             out_features=2)

        self.fc3 = nn.Linear(in_features=fc3_sizes.in_features, out_features=fc3_sizes.out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.fc1_sizes.in_features)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
