import librosa
import numpy as np


def load_audio(path, sr, mono=False):
    data, sanity_sr = librosa.load(path, sr=sr, mono=mono)

    if len(data.shape) == 1:
        data = data[np.newaxis, :]
    return data, sanity_sr
