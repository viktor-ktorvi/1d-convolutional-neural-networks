import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from network_model import Network
from utils.signals import generateSignalData
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
    num_epochs = 10
    batch_size = 64
    lr = 0.003
    holdout_ratio = 0.7

    train_num = round(holdout_ratio * num_signals)
    test_num = num_signals - train_num

    # %% Generate data
    ground_truth, signal_labels, signal_data = generateSignalData(num_signals=num_signals,
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

    dataset = TensorDataset(torch.tensor(signal_data), torch.tensor(signal_labels).type(torch.LongTensor))

    # holdout
    train_set, test_set = torch.utils.data.random_split(dataset, [train_num, test_num])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=test_num)
    # %% Training
    torch.manual_seed(26)

    model = Network(signal_len, classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(model)

    start = time.time()
    print("Training started")

    loss_array = []
    for epoch in range(num_epochs):
        running_loss = 0

        for i, data in enumerate(train_dataloader, 0):
            test_signals, test_labels = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float)

            # reformating the label array to the form that the loss expects
            test_labels = test_labels.view(test_labels.shape[0])
            test_labels = test_labels.type(torch.LongTensor)

            optimizer.zero_grad()

            outputs = model(test_signals.unsqueeze(1) / data_std)
            loss = criterion(outputs, test_labels)
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
            outputs = model(test_signals.unsqueeze(1))

            _, predicted = torch.max(outputs, 1)

    predicted = predicted.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy().reshape(test_num)

    results = predicted == test_labels

    print("\nAccurracy = %.3f %%" % (np.sum(results) / len(results) * 100))

    cm = confusion_matrix(test_labels, predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Greens')

    # TODO see the examples that the network got wrong

