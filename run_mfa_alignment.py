#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniil Kocharov (dan_ya)
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from time import time


def parse_args(parser):
    parser.add_argument('--data-dir', type=str, default=os.path.join('..', '..', 'Corpora', 'synthetic_speech'), help='Data directory')
    parser.add_argument('--datasets', type=str, default='dataset_alignments.json', help='JSON with list of datasets to be processed.')
    # Pipeline parameters
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite data files created before.')
    parser.add_argument('--create-tar', action='store_true', default=False, help='Store the result as TAR file (True). Store as directory with aligments, if False')
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
        text_path = Path(args.data_dir, data_paths[i]['text'])

        output_dir = Path(args.data_dir, audio_path.with_suffix('.aligned'))
        if not args.overwrite and output_dir.is_dir():
            continue

        assert audio_path.is_dir() or audio_path.is_file(), f"The data path doesn't exist: {audio_path}"
        assert text_path.is_dir() or text_path.is_file(), f"The data path doesn't exist: {text_path}"

        tmp_dir = Path(args.data_dir, audio_path.with_suffix('.tmp_mfa'))
        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        print(' - copying audio...')
        if audio_path.suffix == '.tar':
            with tarfile.open(audio_path, 'r') as tar:
                tar.extractall(path=tmp_dir)
        else:
            shutil.copytree(audio_path, tmp_dir, dirs_exist_ok=True)
        print(' - copying text...')
        if text_path.is_dir():
            shutil.copytree(text_path, tmp_dir, dirs_exist_ok=True)
        else:
            text_data = read_text_data(text_path)
            for unit in text_data:
                text_content = unit[0].upper()
                file_path = Path(tmp_dir, Path(unit[1]).with_suffix('.lab'))
                with open(file_path, 'w', encoding='utf-8') as fo:
                    fo.write(text_content)

        # Problem: it is not printing out the messages from system_command.
        # mfa align -j 1 --fast_textgrid_export INPUT_DIR G2P_DICT AC_MODEL OUTPUT_DIR
        print(' - MFA running...')
        system_command = f'conda run --name TORCH bash -c "mfa align --final_clean -j 1 --fast_textgrid_export {tmp_dir} english_us_mfa english_mfa {output_dir}"'
        process = subprocess.Popen(system_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())

        shutil.rmtree(tmp_dir)

        if args.create_tar:
            with tarfile.open(output_dir.with_suffix('.aligned.tar'), "w") as tar:
                for file in output_dir.iterdir():
                    tar.add(file, arcname=file.name)
            shutil.rmtree(output_dir)

        processing_time = time() - timestamp
        hours, remainder = divmod(processing_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f" - h:m:s {int(hours)}:{int(minutes)}:{round(seconds,2)}")
        timestamp = time()


def read_text_data(path: Path) -> list[list[str]]:
    with open(path, 'r', encoding='utf-8') as fi:
        data = fi.readlines()
        data = [line.strip().split('\t') for line in data]
    return data


if __name__ == '__main__':
    main()
