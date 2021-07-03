import numpy as np


def getSignal(wave, amp, freq, phase, t):
    return amp * wave(2 * np.pi * freq * t + phase)


def generateSignalData(num_signals, signal_len, classes, waves, amp_max, amp_min, freq_max, freq_min, t,
                       noise_std_percent):
    signal_data = np.zeros((num_signals, signal_len))
    noise = np.zeros((num_signals, signal_len))
    cathegorical_labels = np.zeros((num_signals, 1))

    # make a signal from a random class with random parameters
    for i in range(num_signals):
        chooser = np.random.randint(len(classes))
        cathegorical_labels[i] = chooser

        # uniformally pick parameters
        amp = np.random.rand() * (amp_max - amp_min) + amp_min
        freq = np.random.rand() * (freq_max - freq_min) + freq_min
        phase = np.random.rand() * 2 * np.pi

        # awgn for good measure
        noise_std = noise_std_percent * amp

        signal_data[i, :] = getSignal(wave=waves[chooser],
                                      amp=amp,
                                      freq=freq,
                                      phase=phase,
                                      t=t).reshape(1, signal_len)

        noise[i, :] = noise_std * np.random.randn(1, signal_len)

    noisy_data = signal_data + noise

    return signal_data, cathegorical_labels, noisy_data
