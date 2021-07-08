import numpy as np
import scipy.signal as signal
from utils.signals import getSignal
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # %% Signal parameters

    classes = ["sin", "square", "saw"]

    waves = [np.sin, signal.square, signal.sawtooth]

    Fs = 2000
    signal_len = 200
    t = np.linspace(0, (signal_len - 1) / Fs, signal_len)
    amp_max = 10
    amp_min = 0
    freq_max = 100
    freq_min = 10

    noise_std_percent = 0.1

    filter_len = 5  # odd number

    fig, ax = plt.subplots(3, 1)
    for i in range(3):
        amp = np.random.rand() * (amp_max - amp_min) + amp_min
        freq = np.random.rand() * (freq_max - freq_min) + freq_min
        phase = np.random.rand() * 2 * np.pi

        # %% Generating the signal
        x = getSignal(wave=waves[i],
                      amp=amp,
                      freq=freq,
                      phase=phase,
                      t=t)
        xnoise = x + noise_std_percent * amp * np.random.randn(signal_len)

        # %% Anticausal FIR filter
        b = np.ones(filter_len)
        a = filter_len
        xfiltered = signal.filtfilt(b, a, xnoise)
        ax[i].plot(xnoise, label='signal + noise')
        ax[i].plot(xfiltered, color='r', label='filtered signal')
        ax[i].set_title(classes[i])
        ax[i].set_xlabel('n [sample]')
        ax[i].set_ylabel('x(n) [unit]')
        ax[i].legend()
    fig.suptitle("Filtered signals", horizontalalignment='center')
    plt.tight_layout()
    plt.show()
