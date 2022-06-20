import os
import argparse
from functools import partial
import time

import torch
import numpy as np
from tqdm import tqdm

from model.SSFou import SSFou
import model.utils as model_utils
from data.utils import random_amplify
from data.dataset import SF_Dataset
from utils.get_musdbhq import get_musdbhq


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

    augment_func = partial(random_amplify, p_min=0.3, p_max=1.3)

    train_data = SF_Dataset(
        musdb["train"],
        args.hdf_dir,
        args.hdf_filename + "_train",
        ssfou.aether,
        mode="shuffle",
        audio_transform=augment_func,
        sr=args.sr,
    )
    val_data = SF_Dataset(
        musdb["val"],
        args.hdf_dir,
        args.hdf_filename + "_val",
        ssfou.aether,
        mode="static",
        sr=args.sr,
    )
    test_data = SF_Dataset(
        musdb["test"],
        args.hdf_dir,
        args.hdf_filename + "_test",
        ssfou.aether,
        mode="static",
        sr=args.sr,
    )
    dataloader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(params=ssfou.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    state = {"steps": 0, "epochs": 0, "best loss": np.inf}
    if args.cuda:
        # ssfou = model_utils.DataParallel(ssfou)
        ssfou.cuda()
    if args.load_model is not None:
        print(f"Continuing training from checkpoint {args.load_model}")
        state = model_utils.load_model(ssfou, optimizer, args.load_model, args.cuda)

    # Training
    print("Training Start")
    _epochs = np.inf if args.epochs is None else args.epochs
    while state["epochs"] < _epochs:
        if train_data.mode == "shuffle":
            if state["epochs"] % args.shuffle_freq == 0:
                print("Shuffling dataset")
                train_data.shuffle()
                dataloader = torch.utils.data.DataLoader(
                    train_data, args.batch_size, shuffle=True
                )

        avg_loss = 0
        print(f"Training from epoch : {state['epochs']}, step : {state['steps']}")
        ssfou.train()
        with tqdm(total=len(train_data) // args.batch_size) as pbar:
            np.random.seed()
            for idx, (mixture, source) in enumerate(dataloader):
                if args.cuda:
                    mixture = mixture.cuda()
                    for key in list(source.keys()):
                        source[key] = source[key].cuda()

                t = time.time()
                optimizer.zero_grad()
                total_loss = 0

                output = ssfou(mixture)
                for inst in ssfou.aether["instruments"]:
                    loss = criterion(output[inst], source[inst])
                    loss.backward()
                    total_loss += loss.item()
                optimizer.step()
                state["steps"] += 1
                avg_loss = avg_loss + (
                    total_loss / len(ssfou.aether["instruments"]) - avg_loss
                ) / (idx + 1)
                pbar.update(1)
        state["epochs"] += 1
        print(f"Training average loss : {avg_loss}")

        if state["epochs"] % args.validation_freq == 0:
            ssfou.eval()
            val_dataloader = torch.utils.data.DataLoader(
                val_data, args.batch_size, shuffle=False
            )
            avg_loss = 0
            with tqdm(total=len(val_data) // args.batch_size) as pbar:
                for idx, (mixture, source) in enumerate(val_dataloader):
                    if args.cuda:
                        mixture = mixture.cuda()
                        for key in list(source.keys()):
                            source[key] = source[key].cuda()

                    total_loss = 0
                    output = ssfou(mixture)

                    for inst in ssfou.aether["instruments"]:
                        loss = criterion(output[inst], source[inst])
                        total_loss += loss.item()
                    avg_loss = avg_loss + (
                        total_loss / len(ssfou.aether["instruments"]) - avg_loss
                    ) / (idx + 1)
                    pbar.set_description(f"current_loss : {avg_loss}")
                    pbar.update(1)
        if state["epochs"] % args.saving_freq == 0:
            model_utils.save_model(ssfou, optimizer, state, os.path.join(args.checkpoint_dir, f"checkpoint_{state['epochs']}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument(
        "--dataset_dir", required=True, type=str, help="Path to musdb18hq dataset"
    )
    parser.add_argument(
        "--hdf_dir", required=True, type=str, help="Path to hdf directory"
    )
    parser.add_argument(
        "--hdf_filename", type=str, default="SS-Fou", help="Namespace for hdf_file"
    )
    parser.add_argument(
        "--cuda", action="store_true", help="Use CUDA (default : False)"
    )
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save the checkpoints in",
    )
    parser.add_argument(
        "--load_model", type=str, default=None, help="Path to checkpoint to load"
    )

    # model
    parser.add_argument(
        "--features",
        type=int,
        default=16,
        help="Number of base feature channels per layer",
    )
    parser.add_argument(
        "--feature_growth",
        type=str,
        default="double",
        help="the feature growth of the amount of features, has to be one of ('double', 'add', 'constant')",
    )
    parser.add_argument(
        "--levels", type=int, default=10, help="Number of convolution layers"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=2,
        help="Number of audio channels, 1 for mono, 2 for stereo",
    )
    parser.add_argument(
        "--n_fft", type=int, default=11, help="Exponent of 2, the size of the fft frame"
    )
    parser.add_argument(
        "--hop_factor",
        type=int,
        default=2,
        help="Exponent of (1/2), fraction of n_fft for hop size",
    )
    parser.add_argument("--frames", type=int, default=25, help="Amount of frames stft")
    parser.add_argument(
        "--kernel_size", type=int, default=5, help="Kernel Size for convolution"
    )

    # training
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--shuffle_freq",
        type=int,
        default=10,
        help="frequency of shuffling the dataset when using mode=shuffle",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Amount of epochs the model is to be trained for, don't specify for infinite training",
    )
    parser.add_argument(
        "--saving_freq",
        type=int,
        default=1,
        help="Frequency of saving the model (epochs)",
    )
    parser.add_argument(
        "--validation_freq",
        type=int,
        default=1,
        help="Frequency of model validation (epochs",
    )

    args = parser.parse_args()
    main(args)
