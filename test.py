import os
import argparse
from functools import partial
import time
import pickle

import torch
import numpy as np
from tqdm import tqdm
import museval

from model.SSFou import SSFou
import model.utils as model_utils
from data.utils import random_amplify, load_audio
from data.dataset import SF_Dataset
from utils.get_musdbhq import get_musdbhq
from torch.utils.tensorboard import SummaryWriter
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

    musdb = get_musdbhq(args.dataset_dir)
    writer = SummaryWriter(args.log_dir)

    optimizer = torch.optim.Adam(params=ssfou.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    state = {"steps": 0, "epochs": 0, "best loss": np.inf}
    if args.cuda:
        # ssfou = model_utils.DataParallel(ssfou)
        ssfou.cuda()

    testing_epochs = [i * 10 for i in range(2, 13)]
    for _ep in testing_epochs:
        load_model = os.path.join(args.checkpoint_dir, f"checkpoint_{_ep}")
        print(f"testing checkpoint {load_model}")
        state = model_utils.load_model(ssfou, optimizer, load_model, args.cuda)

        ssfou.eval()
        avg_loss = 0
        perfs = list()
        with tqdm(total=len(musdb["test"])) as pbar:
            for idx, song in enumerate(musdb["test"]):
                preds = predict_song(ssfou, song["mixture"], cuda=args.cuda)

                preds = np.stack(preds[inst].T for inst in ssfou.aether["instruments"])
                targets = np.stack(
                    [load_audio(song[inst])[0].T for inst in ssfou.aether["instruments"]]
                )
                
                SDR, ISR, SIR, SAR, _ = museval.metrics.bss_eval(targets, preds)


                perf = dict()
                for idx, inst in enumerate(ssfou.aether["instruments"]):
                    perf[inst] = {
                        "SDR" : SDR[idx],
                        "ISR" : ISR[idx],
                        "SIR" : SIR[idx],
                        "SAR" : SAR[idx]
                    }
                perfs.append(perf)
                pbar.update(1)
        with open(os.path.join(args.checkpoint_dir, f"results_{_ep}.pkl"), "wb") as f:
            pickle.dump(perfs, f)

        avg_SDR = {
            inst : np.mean([np.nanmean(song[inst]["SDR"]) for song in perfs]) for inst in ssfou.aether["instruments"]
        }
        avg_ISR = {
            inst : np.mean([np.nanmean(song[inst]["ISR"]) for song in perfs]) for inst in ssfou.aether["instruments"]
        }
        avg_SIR = {
            inst : np.mean([np.nanmean(song[inst]["SIR"]) for song in perfs]) for inst in ssfou.aether["instruments"]
        }
        avg_SAR = {
            inst : np.mean([np.nanmean(song[inst]["SAR"]) for song in perfs]) for inst in ssfou.aether["instruments"]
        }

        for inst in ssfou.aether["instruments"]:
            writer.add_scalar(f"TEST_SDR_{inst}", avg_SDR[inst], _ep)
            writer.add_scalar(f"TEST_ISR_{inst}", avg_ISR[inst], _ep)
            writer.add_scalar(f"TEST_SIR_{inst}", avg_SIR[inst], _ep)
            writer.add_scalar(f"TEST_SAR_{inst}", avg_SAR[inst], _ep)

if __name__ == "__main__":
    class Arg:
        def __init__(self):
            self.dataset_dir = os.path.join("/","home","guest01","dataset")
            self.hdf_dir = os.path.join(self.dataset_dir, "hdf")
            self.hdf_filename = "SS-Fou"
            self.cuda = True
            self.sr = 44100
            self.checkpoint_dir = os.path.join("/", "home", "guest01", "research", "SS-fou", "checkpoints_Final_shuffle")
            self.load_model = os.path.join(self.checkpoint_dir, "checkpoint_50")
            self.log_dir = os.path.join("/", "home", "guest01", "research", "SS-fou", "logs_Final_shuffle")
 
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
    args = Arg()

    main(args)
