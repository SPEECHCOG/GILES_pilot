#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniil Kocharov (dan_ya)
"""
from argparse import ArgumentParser
import json
import numpy as np
import os
import parselmouth
from pathlib import Path
import pickle
import shutil
import tarfile
from time import time
from tqdm import tqdm


def parse_args(parser):
    parser.add_argument('--data-dir', type=str, default=Path('..', '..', 'Corpora', 'synthetic_speech'), help='Data directory')
    parser.add_argument('--datasets', type=str, default='dataset_paths.json', help='JSON with list of datasets to be processed.')
    # Pipeline parameters
    parser.add_argument('--recalculate', action='store_true', default=False, help='Overwrite data files created before.')
    # Formant detection parameters
    parser.add_argument('--frame-length', type=float, default=0.025, help='The frame length of formant detection (seconds).')
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
        output_file = audio_path.with_suffix('.formant')
        tmp_dir = audio_path.with_suffix('.tmp_formant')
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
                tmp_dir = audio_path

        print(' - calculate formants...')
        audio_files = list(tmp_dir.glob("*.wav")) + list(tmp_dir.glob("*.flac"))
        audio_files = [f for f in audio_files if f.stem[0] not in '.~']
        data_formants = {}
        for file in tqdm(sorted(audio_files)):
            data_formants[file.stem] = get_formants(file)
        data = {'frame_length': args.frame_length, 'data': data_formants}
        with open(output_file, 'wb') as fout:
            pickle.dump(data, fout)
        shutil.rmtree(tmp_dir)

        processing_time = time() - timestamp
        hours, remainder = divmod(processing_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f" - h:m:s {int(hours)}:{int(minutes)}:{round(seconds,2)}")
        timestamp = time()
    return True


def get_formants(file):
    sound = parselmouth.Sound(str(file))
    formants = parselmouth.praat.call(sound, "To Formant (burg)", 0.0, 5.0, 5500, args.frame_length, 50.0)
    ts = np.array([t for t in formants.ts()])
    f1 = np.array([formants.get_value_at_time(1, t) for t in formants.ts()]).astype(np.int16)
    f2 = np.array([formants.get_value_at_time(2, t) for t in formants.ts()]).astype(np.int16)
    data = {'f1': f1, 'f2': f2, 'timestamps': ts}
    return data


if __name__ == '__main__':
    main()
