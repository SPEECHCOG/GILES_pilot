#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniil Kocharov (dan_ya)
"""
from argparse import ArgumentParser
import json
import io
import librosa
from pathlib import Path
import pickle
import tarfile
from time import time
from tqdm import tqdm
import wave


def parse_args(parser):
    parser.add_argument('--data-dir', type=str, default=Path('..', '..', 'Corpora', 'synthetic_speech'), help='Data directory')
    parser.add_argument('--datasets', type=str, default='dataset_paths.json', help='JSON with list of datasets to be processed.')
    # Pipeline parameters
    parser.add_argument('--recalculate', action='store_true', default=False, help='Overwrite data files created before.')
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
        output_file = audio_path.with_suffix('.duration')
        if not args.recalculate and output_file.is_file():
            continue
        assert audio_path.is_file() or audio_path.is_dir(), f"The data path doesn't exist: {audio_path}"
        if audio_path.is_file():
            assert audio_path.suffix == '.tar', f"The audio file is not TAR file: {audio_path}"

        if audio_path.is_file():
            data = {'data': get_durations_from_tar(audio_path)}
        else:
            audio_files = list(audio_path.glob("*.wav")) + list(audio_path.glob("*.flac"))
            audio_files = sorted([f for f in audio_files if f.stem[0] not in '.~'])
            data = {'data': {}}
            for file in tqdm(audio_files):
                data['data'][file.stem] = librosa.get_duration(path=file)
        with open(output_file, 'wb') as fout:
            pickle.dump(data, fout)

        total_duration = sum([data['data'][f] for f in data['data']])
        hours, remainder = divmod(total_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f'dataset {audio_path.stem}: {int(hours)}:{int(minutes)}:{round(seconds,2)}')

        processing_time = time() - timestamp
        hours, remainder = divmod(processing_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f" - h:m:s {int(hours)}:{int(minutes)}:{round(seconds,2)}")
        timestamp = time()


def get_durations_from_tar(path: Path) -> dict[str, float]:
    data = {}
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith('.wav'):
                file_obj = tar.extractfile(member)
                if file_obj:
                    wav_bytes = file_obj.read()
                    wav_file = io.BytesIO(wav_bytes)
                    with wave.open(wav_file, 'rb') as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        duration = frames / float(rate)
                    data[Path(member.name).stem] = duration
    return data


if __name__ == '__main__':
    main()
