#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniil Kocharov (dan_ya)
"""
from argparse import ArgumentParser
import json
import librosa
import numpy as np
import os
from pathlib import Path
import pickle
import shutil
import tarfile
from time import time
from tqdm import tqdm
from typing import Union


def parse_args(parser):
    parser.add_argument('--data-dir', type=str, default=Path('..', '..', 'Corpora', 'synthetic_speech'), help='Data directory')
    parser.add_argument('--datasets', type=str, default='dataset_paths.json', help='JSON with list of datasets to be processed.')
    # Pipeline parameters
    parser.add_argument('--recalculate', action='store_true', default=False, help='Overwrite data files created before.')
    # PDA parameters
    parser.add_argument('--frame-length', type=float, default=0.1, help='The frame length of F0 detection (seconds). Thus the step is 4 times less.')
    return parser.parse_args()


arg_parser = ArgumentParser(description='Calculate prosodic measures of the dataset', allow_abbrev=False)
args = parse_args(arg_parser)


def main():
    timestamp = time()
    with open(args.datasets, 'r', encoding='utf-8') as fin:
        data_paths = json.load(fin)
    for i in range(len(data_paths)):
        print(data_paths[i])
        audio_path = Path(args.data_dir, data_paths[i]['audio'])
        tmp_dir = audio_path.with_suffix('.tmp_f0')
        output_file = audio_path.with_suffix('.f0')
        if not args.recalculate and output_file.is_file():
            continue
        assert audio_path.is_dir() or audio_path.is_file(), f"The data path doesn't exist: {audio_path}"

        print(' - loading audio files...')
        if not tmp_dir.is_dir():
            os.makedirs(tmp_dir, exist_ok=True)
            if audio_path.suffix == '.tar':
                with tarfile.open(audio_path, 'r') as tar:
                    tar.extractall(path=tmp_dir)
            else:
                output_file = audio_path.with_name(f'{audio_path.name}.f0')
                tmp_dir = audio_path

        print(' - calculate f0...')
        audio_files = list(tmp_dir.glob("*.wav")) + list(tmp_dir.glob("*.flac"))
        audio_files = [f for f in audio_files if f.stem[0] not in '.~']
        data_f0 = {}
        for file in tqdm(audio_files):
            data_f0[file.stem] = get_f0(file)

        data = {'frame_length': args.frame_length / 4, 'data': data_f0}
        with open(output_file, 'wb') as fout:
            pickle.dump(data, fout)
        shutil.rmtree(tmp_dir)

        processing_time = time() - timestamp
        hours, remainder = divmod(processing_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f" - h:m:s {int(hours)}:{int(minutes)}:{round(seconds,2)}")
        timestamp = time()


def get_f0(file: Path) -> dict[str, Union[float, np.ndarray]]:
    y, sr = librosa.load(file)
    frame = int(sr * args.frame_length)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, sr=sr, fmin=100, fmax=450, frame_length=frame)
    data = {'f0': f0.astype(int), 'voiced': voiced_flag, 'voiced_prob': voiced_probs.astype(np.float16)}
    return data


if __name__ == '__main__':
    main()
