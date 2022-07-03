import os
import argparse
from functools import partial
import time
import pickle
import soundfile

import torch
import numpy as np
from tqdm import tqdm

from model.SSFou import SSFou
import model.utils as model_utils
from data.utils import random_amplify, load_audio
from data.dataset import SF_Dataset
from predict_song import predict_song


def main(args):
    assert args.levels >= 2
    assert args.channels > 0
    assert args.feature_growth in ["double", "add", "constant"]

    l_features = [args.channels for i in range(args.levels)]
    if args.feature_growth == "double":
        _levels = args.levels - 2
        _mid = (_levels - 1) / 2
        l_features[1:-1] = [
            args.features * 2 ** int(_mid - abs(_mid - x)) for x in range(_levels)
        ]
    elif args.feature_growth == "add":
        _levels = args.levels - 2
        _mid = (_levels - 1) / 2
        l_features[1:-1] = [
            args.features * int(_mid - abs(_mid - x) + 1) for x in range(_levels)
        ]
    elif args.feature_growth == "constant":
        l_features[1:-1] = [args.features for i in range(args.levels - 2)]
    ssfou = SSFou(
        l_features, args.n_fft, args.hop_factor, args.frames, args.kernel_size
    )

    print(f"Model : {ssfou.aether}")
    print(f"Parameter Count : {sum(p.numel() for p in ssfou.parameters())}")

    optimizer = torch.optim.Adam(params=ssfou.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    state = {"steps": 0, "epochs": 0, "best loss": np.inf}
    if args.cuda:
        # ssfou = model_utils.DataParallel(ssfou)
        ssfou.cuda()

    print(f"testing checkpoint {args.load_model}")
    state = model_utils.load_model(ssfou, optimizer, args.load_model, args.cuda)

    ssfou.eval()
    preds = predict_song(ssfou, args.song_path, cuda=args.cuda)
    for inst in ssfou.aether["instruments"]:
        soundfile.write(f"{args.song_path}_{inst}.wav", preds[inst].T, 44100, "PCM_16")


if __name__ == "__main__":

    class Arg:
        def __init__(self):
            self.dataset_dir = os.path.join("/", "home", "guest01", "dataset")
            self.hdf_dir = os.path.join(self.dataset_dir, "hdf")
            self.hdf_filename = "SS-Fou"
            self.cuda = True
            self.sr = 44100
            self.checkpoint_dir = os.path.join(
                "/",
                "home",
                "capu",
                "Projects",
                "SS-Fou",
                "checkpoints",
            )
            self.load_model = os.path.join(self.checkpoint_dir, "checkpoint_120")
            self.log_dir = os.path.join(
                "/", "home", "guest01", "research", "SS-fou", "logs_Final_shuffle"
            )

            self.features = 64
            self.feature_growth = "double"
            self.levels = 8
            self.channels = 2
            self.n_fft = 10
            self.hop_factor = 1
            self.frames = 15
            self.kernel_size = 5

            self.batch_size = 17
            self.lr = 0.0001832404
            self.shuffle_freq = 2
            self.epochs = 150
            self.saving_freq = 5
            self.validation_freq = 1

            self.song_path = "dramaturgy.mp3"

    args = Arg()

    main(args)
