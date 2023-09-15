import random
from typing import List, Callable, Tuple

import hydra
import scipy
import torch

import numpy as np
import torch.nn as nn

from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import R2Score
from torchvision.transforms import Normalize
from tqdm import tqdm

from matplotlib import pyplot as plt

from model import TimeseriesAutoencoder

from utils.signals import generate_noisy_signal


def generate_denoising_autoencoder_data(signals_functions: List[Callable], class_probabilities: List[float], cfg: DictConfig) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Generate data for a denoising autoencoder.

    :param signals_functions: List of signal function to generate.
    :param class_probabilities: List that defines the probability of generating the corresponding class.
    :param cfg: Config.

    :return: Noisy signals, clean signals.
    """
    noisy_signals: List[Tensor] = []
    clean_signals: List[Tensor] = []
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
        clean_signals.append(torch.tensor(clean_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0))

    return noisy_signals, clean_signals


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    # set random seeds
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    # generate data
    signals_functions = [np.sin, scipy.signal.square, scipy.signal.sawtooth, scipy.special.sinc]
    class_probabilities = [0.25, 0.25, 0.25, 0.25]
    noisy_signals, clean_signals = generate_denoising_autoencoder_data(signals_functions, class_probabilities, cfg)

    # split
    noisy_train, noisy_val, clean_train, clean_val = train_test_split(noisy_signals, clean_signals, test_size=cfg.data.validation_split, shuffle=True)

    plot_data(
        noisy_data=noisy_train,
        width=3, height=2,
        suptitle="Training data"
    )
    plt.pause(0.001)

    # datasets and loaders
    dataset_train = TensorDataset(torch.vstack(noisy_train), torch.vstack(clean_train).squeeze())
    dataset_val = TensorDataset(torch.vstack(noisy_val), torch.vstack(clean_val).squeeze())

    loader_train = DataLoader(dataset_train, batch_size=cfg.model.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=cfg.model.batch_size)

    # model and training essentials
    device = torch.device(cfg.model.device if torch.cuda.is_available() else "cpu")

    model = TimeseriesAutoencoder(input_length=cfg.data.signal_length,
                                  hidden_channels=cfg.model.hidden_channels,
                                  kernel_sizes=cfg.model.kernel_sizes,
                                  strides=cfg.model.strides,
                                  dilations=cfg.model.dilations,
                                  paddings=cfg.model.paddings,
                                  input_channels=noisy_train[0].shape[1])
    model.to(device)

    print(model)

    for conv in model.conv:
        print(f"{conv.input_length}->{conv.output_length}")

    for tr_conv in model.transpose_conv:
        print(f"{tr_conv.input_length}->{tr_conv.output_length}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    criterion = nn.MSELoss()

    r2_train = R2Score(num_outputs=cfg.data.signal_length).to(device)
    r2_val = R2Score(num_outputs=cfg.data.signal_length).to(device)

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

            r2_train(out, y)
            optimizer.step()

        # validation loop
        model.eval()
        with torch.no_grad():
            for batch in loader_val:
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                out = model(x)

                r2_val(out, y)

        progress_bar.set_description(f"Training | Validation: r2 score: {r2_train.compute(): 5.3f} | {r2_val.compute(): 5.3f}")
        r2_train.reset()
        r2_val.reset()

    # evaluate
    evaluation_noisy = []
    evaluation_clean = []
    predictions = []

    model.eval()
    with torch.no_grad():
        for batch in loader_val:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            out = model(x)

            evaluation_noisy.append(x.cpu())
            evaluation_clean.append(y.cpu())
            predictions.append(out.cpu())

    # collect all the validation predictions
    evaluation_noisy = torch.vstack(evaluation_noisy)
    evaluation_clean = torch.vstack(evaluation_clean)
    predictions = torch.vstack(predictions)

    plot_data(
        noisy_data=evaluation_noisy,
        predictions=predictions,
        width=3, height=2,
        suptitle="Evaluation"
    )

    plt.show()


def plot_data(noisy_data: Tensor,
              predictions: Tensor = None,
              width: int = 2,
              height: int = 2,
              suptitle: str = ""):
    """
    Plot a random subset of data.

    :param noisy_data:
    :param predictions:
    :param width:
    :param height:
    :param suptitle:
    :return:
    """

    random_idx = np.random.choice(len(noisy_data), size=width * height)  # sample random data points

    fig, axs = plt.subplots(height, width)
    plt.suptitle(suptitle)
    for i in range(width * height):
        ax = axs[i // width, i % width]  # TODO support w or h === 1

        # ax.xaxis.set_tick_params(labelbottom=False)
        # ax.yaxis.set_tick_params(labelleft=False)
        #
        # ax.set_xticks([])
        # ax.set_yticks([])

        ax.set_xlabel("n [sample]")

        idx = random_idx[i]

        ax.plot(noisy_data[idx].squeeze(), color="C0", label="noisy")

        if predictions is not None:
            ax.plot(predictions[idx].squeeze(), color="red", label="pred")
            ax.legend(loc="upper right")


if __name__ == "__main__":
    main()
