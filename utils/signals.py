import numpy as np


def getSignal(wave, amp, freq, phase, t):
    return amp * wave(2 * np.pi * freq * t + phase)
