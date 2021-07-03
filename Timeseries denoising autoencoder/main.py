import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from network_model import Network
from utils.signals import getSignal
import time

if __name__ == "__main__":
    # %% Signal parameters

    # ["sin", "square", "saw"]
    classes = ["sin", "square", "saw"]

    # [np.sin, signal.square, signal.sawtooth]
    waves = [np.sin, signal.square, signal.sawtooth]

    Fs = 2000
    signal_len = 200
    t = np.linspace(0, (signal_len - 1) / Fs, signal_len)
    amp_max = 10
    amp_min = 0
    freq_max = 100
    freq_min = 10

    noise_std_percent = 0.1
    # %% Training parameters
    num_signals = 10000
    num_epochs = 30
    batch_size = 64
    lr = 0.003
    holdout_ratio = 0.7

    train_num = round(holdout_ratio * num_signals)
    test_num = num_signals - train_num

    # %% Generate data

    signal_data = np.zeros((num_signals, signal_len))
    signal_labels = np.zeros((num_signals, signal_len))

    # make a signal from a random class with random parameters
    for i in range(num_signals):
        chooser = np.random.randint(len(classes))

        # uniformally pick parameters
        amp = np.random.rand() * (amp_max - amp_min) + amp_min
        freq = np.random.rand() * (freq_max - freq_min) + freq_min
        phase = np.random.rand() * 2 * np.pi

        # awgn for good measure
        noise_std = noise_std_percent * amp

        signal_labels[i, :] = getSignal(wave=waves[chooser],
                                        amp=amp,
                                        freq=freq,
                                        phase=phase,
                                        t=t).reshape(1, signal_len)

        signal_data[i, :] = signal_labels[i, :] + noise_std * np.random.randn(1, signal_len)

    data_std = np.std(signal_data)
    # %% Visualizing the data
    plt.figure()
    plt.title("No noise VS noise")
    for i in range(1):
        idx = np.random.randint(num_signals)
        plt.plot(signal_labels[idx, :], label="signal")
        plt.plot(signal_data[idx, :], label="signal + noise")
    plt.xlabel("n [sample]")
    plt.ylabel("signal [num]")
    plt.legend()
    plt.show()

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
    model = Network(signal_len)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    print("Training started")

    loss_array = []
    for epoch in range(num_epochs):
        running_loss = 0

        for i, data in enumerate(train_dataloader, 0):
            test_signals, test_labels = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float)

            optimizer.zero_grad()

            outputs = model(test_signals.unsqueeze(1) / data_std) * data_std
            loss = criterion(outputs, test_labels.view(outputs.shape))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("epoch: %d\tloss: %0.10f" % (epoch, running_loss))
        loss_array.append(running_loss)

    end = time.time()
    print("Training complete. It took %5.2f seconds" % (end - start))

    plt.figure()
    plt.title("Loss")
    plt.xlabel("epoch [num]")
    plt.ylabel("loss [num]")
    plt.plot(loss_array)
    plt.show()
    # %% Testing

    # pass the whole test set thorugh the model and check outputs
    with torch.no_grad():
        for data in test_dataloader:
            test_signals, test_labels = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float)
            outputs = model(test_signals.unsqueeze(1) / data_std) * data_std

            mse = criterion(outputs, test_labels.view(outputs.shape))
            print("test MSE = %.3f" % mse)
    # %% Visually checking results

    plt.figure()
    plt.title("Noisy VS denoised")
    for i in range(1):
        idx = np.random.randint(test_num)
        plt.plot(test_signals[idx, :], label="noisy")
        plt.plot(outputs[idx, :].T, label="denoised")
        plt.plot(test_labels[idx, :], label="ground truth")
    plt.xlabel("n [sample]")
    plt.ylabel("signal [num]")
    plt.legend()
    plt.show()
