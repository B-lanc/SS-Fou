from functools import partial

import numpy as np
from scipy.signal import stft, istft
import torch

from data.utils import load_audio


def predict_song(model, song_path, cuda=False, sr=44100):
    mixture, sr = load_audio(song_path, sr)
    _length = mixture.shape[1]
    _input_size = model.aether["input_size"]

    _stft = partial(
        stft,
        nperseg=model.aether["n_fft"],
        noverlap=model.aether["n_fft"] - model.aether["hop_size"],
        boundary=None,
    )
    _istft = partial(
        istft,
        nperseg=model.aether["n_fft"],
        noverlap=model.aether["n_fft"] - model.aether["hop_size"],
        boundary=False,
    )

    pad_back = mixture.shape[1] % _input_size
    pad_back = 0 if pad_back == 0 else _input_size - pad_back

    if pad_back > 0:
        mixture = np.pad(
            mixture, [(0, 0), (0, pad_back)], mode="constant", constant_values=0.0
        )

    with torch.no_grad():
        sources = {key: np.zeros(mixture.shape) for key in model.aether["instruments"]}
        for _window in range(0, mixture.shape[1], _input_size):
            curr_mixture = mixture[:, _window : _window + _input_size]
            curr_stft = _stft(curr_mixture)[2]
            torch_curr_stft = torch.from_numpy(curr_stft.real).unsqueeze(0)
            if cuda:
                torch_curr_stft = torch_curr_stft.cuda()
            output = model(torch_curr_stft)
            
            for inst in model.aether["instruments"]:
                output[inst] = output[inst].squeeze(0).cpu().numpy()
                output[inst][output[inst] >= 0.5] = 1
                output[inst][output[inst] < 0.5] = 0
                sources[inst][:, _window : _window + _input_size] = _istft(
                    output[inst] * curr_stft
                )[1]
    for inst in model.aether["instruments"]:
        sources[inst] = sources[inst][:, :_length]

    return sources
