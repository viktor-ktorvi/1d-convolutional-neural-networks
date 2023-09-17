import random
from typing import List, Tuple, Callable

import hydra
import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch import nn, Tensor
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm

from model import TimeseriesClassifier
from utils.signals import generate_noisy_signal


def generate_classification_data(signals_functions: List[Callable], class_probabilities: List[float], cfg: DictConfig) -> Tuple[List[Tensor], List[Tensor], List[str]]:
    """
    Generate data for signal classification.

    :param signals_functions: List of signal function to generate.
    :param class_probabilities: List that defines the probability of generating the corresponding class.
    :param cfg: Config.
    :return: Noisy signals, corresponding labels, class names.
    """
    class_names = [func.__name__ for func in signals_functions]

    noisy_signals: List[Tensor] = []
    labels: List[Tensor] = []
    for _ in tqdm(range(cfg.data.num_samples), desc="Creating signals at random"):
        noisy_signal, clean_signal, signal_func, amplitude, frequency, phase = generate_noisy_signal(
            signals_functions,
            signal_length=cfg.data.signal_length,
            signal_duration=cfg.data.signal_duration,
            relative_noise_std=cfg.data.relative_noise_std,
            amplitude_low=cfg.data.amplitude_low,
            amplitude_high=cfg.data.amplitude_high,
            frequency_low=cfg.data.frequency_low,
            frequency_high=cfg.data.frequency_high,
            phase_low=cfg.data.phase_low,
            phase_high=cfg.data.phase_high,
            class_probabilities=class_probabilities
        )

        noisy_signals.append(torch.tensor(noisy_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        labels.append(torch.tensor(class_names.index(signal_func.__name__)))

    return noisy_signals, labels, class_names


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    # set random seeds
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    # generate data
    signals_functions = [np.sin, scipy.signal.square, scipy.signal.sawtooth, scipy.special.sinc]
    class_probabilities = [0.1, 0.3, 0.2, 0.4]
    noisy_signals, labels, class_names = generate_classification_data(signals_functions, class_probabilities, cfg)

    # split
    signals_train, signals_val, labels_train, labels_val = train_test_split(noisy_signals, labels, test_size=cfg.data.validation_split, shuffle=True)

    plot_data(
        data=signals_train,
        true_labels=labels_train,
        width=3, height=2,
        class_names=class_names,
        suptitle="Training data"
    )
    plt.pause(0.001)

    # datasets and loaders
    dataset_train = TensorDataset(torch.vstack(signals_train), torch.vstack(labels_train).squeeze())
    dataset_val = TensorDataset(torch.vstack(signals_val), torch.vstack(labels_val).squeeze())

    loader_train = DataLoader(dataset_train, batch_size=cfg.model.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=cfg.model.batch_size)

    # model and training essentials
    device = torch.device(cfg.model.device if torch.cuda.is_available() else "cpu")

    model = TimeseriesClassifier(input_length=cfg.data.signal_length,
                                 num_classes=len(class_names),
                                 hidden_channels=cfg.model.hidden_channels,
                                 kernel_sizes=cfg.model.kernel_sizes,
                                 strides=cfg.model.strides,
                                 dilations=cfg.model.dilations,
                                 paddings=cfg.model.paddings,
                                 input_channels=signals_train[0].shape[1])
    model.to(device)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # metrics
    accuracy_train = Accuracy(task="multiclass", num_classes=len(class_names)).to(device)
    accuracy_val = Accuracy(task="multiclass", num_classes=len(class_names)).to(device)

    # train
    progress_bar = tqdm(range(cfg.model.num_epochs), desc="Training | Validation")
    for epoch in progress_bar:

        # training loop
        model.train()
        for batch in loader_train:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            accuracy_train(out, y)
            optimizer.step()

        # validation loop
        model.eval()
        with torch.no_grad():
            for batch in loader_val:
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                out = model(x)

                accuracy_val(out, y)

        progress_bar.set_description(f"Training | Validation: accuracy: {accuracy_train.compute(): 5.3f} | {accuracy_val.compute(): 5.3f}")
        accuracy_train.reset()
        accuracy_val.reset()

    # evaluate
    evaluation_data = []
    evaluation_labels = []
    predictions = []

    model.eval()
    with torch.no_grad():
        for batch in loader_val:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            out = model(x)

            evaluation_data.append(x.cpu())
            evaluation_labels.append(y.cpu())
            predictions.append(torch.argmax(out.cpu(), dim=1))

    # collect all the validation predictions
    evaluation_data = torch.vstack(evaluation_data)
    evaluation_labels = torch.hstack(evaluation_labels)
    predictions = torch.hstack(predictions)

    # plot the results
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(evaluation_labels.numpy(), predictions.numpy()),
        display_labels=class_names
    )
    cm_display.plot(cmap='Greens')

    width, height = 5, 4

    model.eval()
    model.to("cpu")

    plot_data(
        data=evaluation_data,
        true_labels=evaluation_labels,
        predictions=predictions,
        width=width, height=height,
        class_names=class_names,
        suptitle="Evaluation"
    )

    plt.show()


def plot_data(data: Tensor,
              true_labels: Tensor,
              class_names: List[str],
              predictions: Tensor = None,
              width: int = 2,
              height: int = 2,
              suptitle: str = ""):
    """
    Plot a random subset of data.

    :param data:
    :param true_labels:
    :param class_names:
    :param predictions:
    :param width:
    :param height:
    :param suptitle:
    :return:
    """

    random_idx = np.random.choice(len(data), size=width * height)  # sample random data points

    fig, axs = plt.subplots(height, width)
    plt.suptitle(suptitle)
    for i in range(width * height):
        idx = random_idx[i]

        plotting_data = data[idx].squeeze()
        true_class_label = f"truth: {class_names[true_labels[idx]]}"
        predicted_class_label = f"pred: {class_names[predictions[idx]]}" if predictions is not None else ""

        if predictions is None:
            color = "C0"  # default blue
        else:
            if true_labels[idx] == predictions[idx]:
                color = "green"
            else:
                color = "red"

        ax = axs[i // width, i % width]

        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.plot(plotting_data, color=color, label=predicted_class_label)
        ax.set_title(true_class_label)

        if predictions is not None:
            ax.legend(loc="upper right")


if __name__ == "__main__":
    main()
