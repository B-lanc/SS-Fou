import os
from tqdm import tqdm
from functools import partial
import random
import warnings

import h5py
import numpy as np
from scipy.signal import stft
from sortedcontainers import SortedList
from torch.utils.data import Dataset
from .utils import load_audio, BM


class SF_Dataset(Dataset):
    """ """

    def __init__(
        self,
        dataset,
        hdf_dir,
        filename,
        aether,
        mode="static",
        audio_transform=None,
        in_memory=False,
        sr=44100,
    ):
        """
        :params dataset: list of dictionary to audio paths
        :params hdf_dir: directory containing hdf file
        :params filename: the filename for the hdf file
        :params aether: data from the model (SSFou_object.aether)
        :params mode: ["static", "random", "shuffle"], mode for getting the data, static for val and test dataset
        :params audio_transform: function that transforms the data
        :params in_memory: load the hdf file into memory
        :params sr: sample rate
        """

        assert mode in ["static", "random", "shuffle"]
        self._mode = mode
        self._audio_transform = audio_transform
        self._input_size = aether["input_size"]
        self._instruments = aether["instruments"]
        self.stft = partial(
            stft,
            nperseg=aether["n_fft"],
            noverlap=aether["n_fft"] - aether["hop_size"],
            boundary=None,
        )

        _hdf = os.path.join(hdf_dir, filename + ".hdf5")
        if not os.path.exists(_hdf):
            if not os.path.exists(hdf_dir):
                os.makedirs(hdf_dir)
            with h5py.File(_hdf, "w") as f:
                f.attrs["sr"] = sr
                f.attrs["channels"] = aether["features"][0]
                f.attrs["instruments"] = aether["instruments"]

                print("Adding data to the hdf file")
                for idx, track in enumerate(tqdm(dataset)):
                    grp = f.create_group(str(idx))
                    for inst in self._instruments:
                        data, sr = load_audio(
                            track[inst],
                            sr=sr,
                            mono=True if aether["features"][0] == 1 else False,
                        )
                        grp.create_dataset(
                            inst, shape=data.shape, dtype=data.dtype, data=data
                        )
                    grp.attrs["length"] = data.shape[1]

        # check if the hdf5 file is the same architecture
        with h5py.File(_hdf, "r") as f:
            if f.attrs["sr"] != sr:
                raise ValueError(
                    "Tried to load existing hdf file, but the sr value is different"
                )
            if f.attrs["channels"] != aether["features"][0]:
                raise ValueError(
                    "Tried to load existing hdf file, but the amount of channels is different"
                )
            if set(f.attrs["instruments"]) != set(aether["instruments"]):
                raise ValueError(
                    "Tried to load existing hdf file, but the instruments list is different"
                )

            _lengths = [f[str(idx)].attrs["length"] for idx in range(len(f))]
        _lengths = [(l // aether["input_size"]) + 1 for l in _lengths]

        self._start_pos = SortedList(np.cumsum(_lengths))
        self._length = self._start_pos[-1]
        _driver = "core" if in_memory else None
        self._hdf_dataset = h5py.File(_hdf, "r", driver=_driver)
        self._shuffle_dict = {
            inst: np.arange(self._length) for inst in self._instruments
        }

    def __getitem__(self, index):
        sources = dict()
        for inst in self._instruments:
            if self._mode == "random":
                source = self._getStem(inst, random.randint(0, self._length))
            elif self._mode == "shuffle":
                source = self._getStem(inst, self._shuffle_dict[inst][index])
            else:
                source = self._getStem(inst, index)
            if self._audio_transform is not None:
                source = self._audio_transform(source)
            sources[inst] = source

        mixture = np.sum(list(sources.values()), axis=0)
        mixture = np.clip(mixture, -1, 1)

        stft_mixture = self.stft(mixture)[2]
        stft_sources = {
            inst: BM(stft_mixture, self.stft(sources[inst])[2]) for inst in sources
        }
        return stft_mixture.real, stft_sources

    def _getStem(self, stem, index):
        audio_idx = self._start_pos.bisect_right(index)
        if audio_idx > 0:
            index = index - self._start_pos[audio_idx - 1]
        audio_length = self._hdf_dataset[str(audio_idx)].attrs["length"]

        start_pos = index * self._input_size
        end_pos = start_pos + self._input_size

        if end_pos > audio_length:
            pad_back = end_pos - audio_length
            end_pos = audio_length
            start_pos = start_pos - pad_back

        source = self._hdf_dataset[str(audio_idx)][stem][:, start_pos:end_pos].astype(
            np.float32
        )
        return source

    def __len__(self):
        return self._length

    def shuffle(self):
        for inst in self._shuffle_dict:
            np.random.shuffle(self._shuffle_dict[inst])

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value in ["static", "random", "shuffle"]:
            self._mode = value
        else:
            raise ValueError(
                "Value for mode must be either 'static', 'random', or 'shuffle'"
            )
