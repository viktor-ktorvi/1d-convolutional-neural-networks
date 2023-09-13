from typing import Callable, Tuple, Union, List

import random
import scipy

import numpy as np

from numpy.typing import NDArray

from matplotlib import pyplot as plt


def generate_signal(signal_func: Callable, amplitude: float, frequency: float, phase: float, time: NDArray) -> NDArray:
    """
    Generate a time signal.
    :param signal_func:
    :param amplitude:
    :param frequency:
    :param phase:
    :param time:
    :return:
    """
    return amplitude * signal_func(2 * np.pi * frequency * time + phase)


def generate_signal_at_random(signals_functions: List[Callable],
                              time: NDArray,
                              amplitude_low: float = 0.1,
                              amplitude_high: float = 10.0,
                              frequency_low: float = 5,
                              frequency_high: float = 15,
                              phase_low: float = 0.0,
                              phase_high: float = 6.28,
                              class_probabilities: Union[List, Tuple, NDArray] = None) -> Tuple[NDArray, Callable, float, float, float]:
    """
    Generate a signal with random parameters.

    :param signals_functions:
    :param time:
    :param amplitude_low:
    :param amplitude_high:
    :param frequency_low:
    :param frequency_high:
    :param phase_low:
    :param phase_high:
    :param class_probabilities:
    :return:
    """
    if class_probabilities is None:
        class_probabilities = np.ones(len(signals_functions)) / 3

    amplitude = np.random.uniform(low=amplitude_low, high=amplitude_high)
    frequency = np.random.uniform(low=frequency_low, high=frequency_high)
    phase = np.random.uniform(low=phase_low, high=phase_high)

    signal_func = random.choices(signals_functions, weights=class_probabilities)[0]

    signal = generate_signal(signal_func=signal_func,
                             amplitude=amplitude,
                             frequency=frequency,
                             phase=phase,
                             time=time)

    return signal, signal_func, amplitude, frequency, phase


def add_white_noise(signal: NDArray, std: float) -> NDArray:
    """
    Add white gaussian noise to a signal.
    :param signal:
    :param std:
    :return:
    """
    noise = np.random.randn(len(signal)) * std

    return signal + noise


def time_axis(signal_length: int = 256,
              signal_duration: float = 1.0, ) -> NDArray:
    """
    Create a time axis using linspace.
    :param signal_length:
    :param signal_duration:
    :return:
    """
    sampling_frequency = signal_length / signal_duration

    return np.linspace(start=0, stop=(signal_length - 1) / sampling_frequency, num=signal_length)


def generate_noisy_signal(signal_functions: List[Callable],
                          signal_length: int = 256,
                          signal_duration: float = 1.0,
                          relative_noise_std: float = 0.1,
                          amplitude_low: float = 0.1,
                          amplitude_high: float = 10.0,
                          frequency_low: float = 5,
                          frequency_high: float = 15,
                          phase_low: float = 0.0,
                          phase_high: float = 6.28,
                          class_probabilities: Union[List, Tuple, NDArray] = None) -> Tuple[NDArray, NDArray, Callable, float, float, float]:
    """
    Generate a signal with random parameters and add noise to it.
    :param signal_functions:
    :param signal_length:
    :param signal_duration:
    :param relative_noise_std:
    :param amplitude_low:
    :param amplitude_high:
    :param frequency_low:
    :param frequency_high:
    :param phase_low:
    :param phase_high:
    :param class_probabilities:
    :return:
    """
    time = time_axis(signal_length, signal_duration)

    clean_signal, signal_func, amplitude, frequency, phase = generate_signal_at_random(signal_functions,
                                                                                       time,
                                                                                       amplitude_low,
                                                                                       amplitude_high,
                                                                                       frequency_low,
                                                                                       frequency_high,
                                                                                       phase_low,
                                                                                       phase_high,
                                                                                       class_probabilities)

    noisy_signal = add_white_noise(clean_signal, amplitude * relative_noise_std)

    return noisy_signal, clean_signal, signal_func, amplitude, frequency, phase


def main():
    # args
    signal_length = 256
    signal_duration = 1

    amplitude_low = 10
    amplitude_high = 1000

    frequency_low = 5
    frequency_high = 20

    phase_low = 0
    phase_high = 2 * np.pi

    relative_noise_std = 0.1

    signals_functions = (np.sin, scipy.signal.square, scipy.signal.sawtooth)
    class_probabilities = None

    time = time_axis(signal_length, signal_duration)

    noisy_signal, clean_signal, signal_func, amplitude, frequency, phase = generate_noisy_signal(signals_functions,
                                                                                                 signal_length,
                                                                                                 signal_duration,
                                                                                                 relative_noise_std,
                                                                                                 amplitude_low,
                                                                                                 amplitude_high,
                                                                                                 frequency_low,
                                                                                                 frequency_high,
                                                                                                 phase_low,
                                                                                                 phase_high,
                                                                                                 class_probabilities)

    plt.figure()
    plt.title(signal_func.__name__)
    plt.xlabel("t [s]")
    plt.plot(time, noisy_signal, label="noisy signal", color="#E27E1D")
    plt.plot(time, clean_signal, label="clean signal", color="#1D81E2", linewidth=3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
