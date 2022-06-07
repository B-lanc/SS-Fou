import librosa
import numpy as np


def load_audio(path, sr, mono=False):
    data, sanity_sr = librosa.load(path, sr=sr, mono=mono)

    if len(data.shape) == 1:
        data = data[np.newaxis, :]
    return data, sanity_sr


def random_amplify(audio_data, p_min=0.0, p_max=1.0):
    return np.clip(audio_data * np.random.uniform(p_min, p_max), -1.0, 1.0)
