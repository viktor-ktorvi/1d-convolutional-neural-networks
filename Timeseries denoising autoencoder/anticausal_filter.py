import hydra
import numpy as np
import scipy
from matplotlib import pyplot as plt

from utils.signals import generate_noisy_signal


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    signals_functions = [np.sin, scipy.signal.square, scipy.signal.sawtooth, scipy.special.sinc]
    classes = [func.__name__ for func in signals_functions]

    filter_len = 3  # odd number

    width, height = 2, 2

    assert width * height == len(signals_functions)

    fig, axs = plt.subplots(height, width)
    for i in range(len(signals_functions)):
        noisy_signal, clean_signal, signal_func, amplitude, frequency, phase = generate_noisy_signal(
            [signals_functions[i]],
            cfg.data.signal_length,
            cfg.data.signal_duration,
            cfg.data.relative_noise_std,
            cfg.data.amplitude_low,
            cfg.data.amplitude_high,
            cfg.data.frequency_low,
            cfg.data.frequency_high,
            cfg.data.phase_low,
            cfg.data.phase_high
        )

        b = np.ones(filter_len)
        a = filter_len
        xfiltered = scipy.signal.filtfilt(b, a, noisy_signal)

        ax = axs[i // width, i % width]

        ax.set_xlabel("n [sample]")
        ax.set_ylabel("y")

        ax.plot(noisy_signal, label='noisy signal')
        ax.plot(xfiltered, color='r', label='filtered signal')
        ax.set_title(classes[i])

        ax.legend(loc="upper right")
    fig.suptitle(
        r"Anticausal averaging filter $\frac{1}{2n + 1}\sum_{i=-n}^{n}z^i$" + f" , n = {filter_len // 2}",
        horizontalalignment='center',
        fontsize=20
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
