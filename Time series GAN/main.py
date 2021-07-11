import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from network_model import Network
from utils.signals import generateSignalData
import time

if __name__ == "__main__":
    # %% Signal parameters

    # ["sin", "square", "saw"]
    classes = ["sin"]

    # [np.sin, signal.square, signal.sawtooth]
    waves = [np.sin]

    Fs = 2000
    signal_len = 200
    t = np.linspace(0, (signal_len - 1) / Fs, signal_len)
    amp_max = 10
    amp_min = 0
    freq_max = 50
    freq_min = 10

    noise_std_percent = 0.1
    # %% Training parameters
    num_signals = 10000
    num_epochs = 100
    batch_size = 64
    lr = 0.003
    holdout_ratio = 0.7

    train_num = round(holdout_ratio * num_signals)
    test_num = num_signals - train_num

    # %% Generate data
    signal_labels, _, signal_data = generateSignalData(num_signals=num_signals,
                                                       signal_len=signal_len,
                                                       classes=classes,
                                                       waves=waves,
                                                       amp_max=amp_max,
                                                       amp_min=amp_min,
                                                       freq_max=freq_max,
                                                       freq_min=freq_min,
                                                       t=t,
                                                       noise_std_percent=noise_std_percent)

    data_std = np.std(signal_data)

    # %% Setting up the data
    device = torch.device("cpu")  # CPU because I have a weak GPU
    print(device)

    dataset = TensorDataset(torch.tensor(signal_data), torch.tensor(signal_labels))

    # holdout
    train_set, test_set = torch.utils.data.random_split(dataset, [train_num, test_num])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=test_num)
    # %% Training
    torch.manual_seed(26)
