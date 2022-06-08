import librosa
import numpy as np


def load_audio(path, sr, mono=False):
    data, sanity_sr = librosa.load(path, sr=sr, mono=mono)

    if len(data.shape) == 1:
        data = data[np.newaxis, :]
    return data, sanity_sr


def random_amplify(audio_data, p_min=0.0, p_max=1.0):
    return np.clip(audio_data * np.random.uniform(p_min, p_max), -1.0, 1.0)


def BM(mixture, source, alpha=1.0, theta=0.5):
    """
    Compute a binary mask of the source based on the mixture
    :params mixture: stft of the mixture
    :params source: stft of the source, same shape as target
    :params alpha: quotient to raise the mixture and source values, higher would mean more sensitive to difference, resulting in more separation
    :params theta: gate to decide the binary value, 1 if the value of source^alpha / mixture^alpha is higher than theta, 0 if lower
    :return: binary mask of the same shape as target and source
    """
    mask = np.divide(
        np.abs(source) ** alpha, np.finfo(float).eps + np.abs(mixture) ** alpha
    )
    mask[mask >= theta] = 1
    mask[mask < theta] = 0

    return mask
