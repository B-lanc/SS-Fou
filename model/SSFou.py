import warnings
import torch
import torch.nn as nn
from torch.nn import functional as F


class SSFou(nn.Module):
    """
    Neural Network model, the NN itself just consists of a bunch of 2D convolutions.
    The input will be shape=(BS, l_features[0], n_frames, 2**n_fft // 2 + 1)
    And the output should be of the same shape

    The output itself will be the estimation of the source's STFT
    """

    def __init__(
        self, l_features, n_fft, n_hopFactor, n_frames, kernel_size, l_instruments=None
    ):
        """
        :params l_features: list of the amount of features (in channel axis) for each layer
        :params n_fft: exponent of 2, size of each stft window to use
        :params n_hopFactor: negative exponent of 2, fraction of stft window for hop size
        :params n_frames: amount of stft frames
        :params kernel_size: kernel size for the convolution
        :params l_instruments: list of instruments, by default it will be ['bass', 'drums', 'other', 'vocals']

        for example, if n_fft is 11, the window size for the stft will be 2048, then if n_hopFactor is 3, the hop size will be 2048 // (2**3) => 256
        Instead of using the size of the input to decide how many frames for the stft, this goes the other way around and decides the size of the input by calculating the n_fft, n_hopFactor, and n_frames, with the formula input_size = 2**n_fft * (1 + (2**(-n_hopFactor)*n_frames))
        """

        super(SSFou, self).__init__()

        # kernel size should be odd
        assert kernel_size % 2 == 1

        if l_instruments == None:
            self.l_instruments = ["bass", "drums", "other", "vocals"]
        else:
            if len(l_instruments) < 2:
                raise ValueError("Instrument list must contain multiple values")
            self.l_instruments = l_instruments
        self.fou = nn.ModuleDict()

        for inst in self.l_instruments:
            _module = nn.ModuleList()
            for i in range(len(l_features) - 1):
                _module.append(
                    nn.Conv2d(
                        l_features[i],
                        l_features[i + 1],
                        kernel_size=kernel_size,
                        padding="same",
                    )
                )
            self.fou[inst] = _module

        self._aether = {
            "features": l_features,
            "n_frames": n_frames,
            "n_fft": 2**n_fft,
            "hop_size": 2**n_fft // (2**n_hopFactor),
            "input_size": int(2**n_fft * (1 + (n_frames - 1) / (2**n_hopFactor))),
            "instruments": self.l_instruments,
        }

    def forward(self, x):
        """ """
        if x.shape[1] != self.aether["features"][0]:
            raise ValueError(
                f"Input data channel ({x.shape[1]}) does not match model channel ({self.aether['features'][0]})"
            )
        if x.shape[2] != self.aether["n_fft"]//2 + 1:
            warnings.warn(
                f"Input data n_frames ({x.shape[3]}) does not match model n_frames ({self.aether['n_fft']}), it should still run, but unexpected behaviors may occur"
            )
        if x.shape[3] != self.aether["n_frames"]:
            warnings.warn(
                f"Input data n_frames ({x.shape[2]}) does not match model n_frames ({self.aether['n_frames']}), it should still run, but unexpected behaviors may occur"
            )

        _out = dict()
        for inst in self.l_instruments:
            _y = x
            for block in self.fou[inst]:
                _y = block(_y)
                _y = F.leaky_relu(_y)
            _out[inst] = _y
        return _out

    @property
    def aether(self):
        return self._aether
