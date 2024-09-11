#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniil Kocharov (dan_ya)
"""
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Iterable
import copy
import json
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pickle
import random
import pandas as pd
import shutil
from scipy.stats import wilcoxon, sem, shapiro, ttest_ind
from scipy.signal import savgol_filter
import seaborn as sns
import tarfile
from tqdm import tqdm
from typing import Callable, Union
from annotation_utils import TextGrid, IntervalTier, Interval

vowel_symbols = {
                 # IPA
                 'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u',
                 'ɪ', 'ʏ', 'ʊ',
                 'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o',
                 'ə',
                 'ɛ', 'œ', 'ɜ', 'ɞ', 'ʌ', 'ɔ',
                 'æ', 'ɐ',
                 'a', 'ɶ', 'ɑ', 'ɒ',
                 # X-SAMPA: https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runMAUSGetInventar?LANGUAGE=eng-US
                 'A', 'a', 'E', 'e', 'I', 'i', 'O', 'o', 'U', 'u',
                 '3', 'V', 'Q', '6', '@', '{'
}
front_vowels = {'i', 'y', 'ɨ', 'ɪ', 'ʏ', 'ɛ', 'æ'}
voiceless_consonant_symbols = {'p', 't', 'k', 's', 'c', 'ç', 'h', 'f', 'θ', 'ʈ', 'ʔ', 'ʃ'}


def parse_args(parser):
    parser.add_argument('--input-path', type=str, default=os.path.join('..', '..', 'Corpora', 'synthetic_speech'), help='Input directory or file path.')
    parser.add_argument('--output-path', type=str, default=os.path.join('..', 'output', 'tts'), help='Output directory path.')
    parser.add_argument('--datasets', type=str, default='dataset_names.json', help='JSON with list of datasets to be processed.')
    parser.add_argument('--analysis-parameter-file', default='analysis_parameters.json', help='JSON with analysis parameters')
    # Pipeline parameters
    parser.add_argument('--sample-data', action='store_true', default=False, help='Sample the data.')
    parser.add_argument('--analyse-data', action='store_true', default=False, help='Analyse the data samples.')
    parser.add_argument('--plot-figures', action='store_true', default=False, help='Plot figures.')
    parser.add_argument('--statistical-tests', action='store_true', default=True, help='Perform statistical tests.')
    parser.add_argument('--plot-comparison', type=str, default='content', choices=['style', 'content'], help='Name of the tier with words.')
    parser.add_argument('--show-figures', action='store_true', default=False, help='Show figures before saving.')
    parser.add_argument('--plot-pdf', action='store_true', default=False, help='Save a figure as a PDF file along with PNG file.')
    # Feature analysis parameters
    parser.add_argument('--random-seed', type=int, default=7, help='Random seed.')
    parser.add_argument('--sample-rate', type=int, default=22050, help='Sample rate.')
    parser.add_argument('--dataset-sample-size', type=float, default=20, help='Size of dataset sample (in hours).')
    parser.add_argument('--f0-smoothing-window', type=int, default=5, help='Smoothing window used for F0 processing.')
    parser.add_argument('--feature-window-length', type=float, default=0.025, help='Window size of feature extraction (in sec.).')
    parser.add_argument('--phone-tier', type=str, default='phones', choices=['MAU', 'phones'], help='Name of the tier with phones.')
    parser.add_argument('--word-tier', type=str, default='words', choices=['ORT-MAU', 'words'], help='Name of the tier with words.')
    parser.add_argument('--relative-values', action='store_true', default=True, help='Calculate relative features (semitones) or linear (Hertz).')
    parser.add_argument('--max-sound-length', type=float, default=0.3, help='Maximal duration of sound (in sec.) to be analyzed.')
    parser.add_argument('--min-vowel-duration', type=float, default=0.05, help='Minimal duration of vowel (in sec.) to be analyzed.')
    parser.add_argument('--outlier-percentile', type=float, default=1.0, help='The percentile to be removed to clean distributions.')
    # Plotting parameters
    parser.add_argument('--show-sem', action='store_true', default=True, help='Plot regions of standard error of mean on linear plots.')
    parser.add_argument('--paper-plots', action='store_true', default=True, help='Plots are prepared for the paper view.')
    parser.add_argument('--plot-distributions', action='store_true', default=False, help='Plot distributions.')
    parser.add_argument('--violin_plot_with_quartiles', action='store_true', default=True, help='Plot distributions as violin plots.')
    parser.add_argument('--font_size', type=int, default=10, help='Size of the font in the plots.')
    return parser.parse_args()


arg_parser = ArgumentParser(description='Calculate prosodic measures of the dataset', allow_abbrev=False)
args = parse_args(arg_parser)


def main():
    with open(args.dataset_file, 'r', encoding='utf-8') as fin:
        datasets_to_compare = json.load(fin)
    dataset_ids = [item for sublist in datasets_to_compare for item in sublist]

    random.seed(args.random_seed)
    if args.paper_plots:
        args.font_size = 16
    report_dir = Path(args.output_path, f'reports')
    figure_dir = Path(args.output_path, f'img')
    sampling_dir = Path(args.output_path, f'samples')
    feature_file = Path(report_dir, f'features.txt')
    statistical_report_file = Path(report_dir, f'statistical_report.txt')
    for d in [report_dir, sampling_dir, figure_dir]:
        os.makedirs(d, exist_ok=True)

    with open(args.analysis_parameter_file, 'r') as fi:
        data = ''.join(line for line in fi if not line.strip().startswith('#'))
        analysis_setup = json.loads(data)

    if args.sample_data:
        print('data sampling...')
        dataset_data = load_raw_data(Path(args.input_path), ds_names=dataset_ids, reload_datasets=True)
        dataset_samples = dataset_sampling(dataset_data, sorted(dataset_data))
        save_dataset_samples(dataset_samples, sampling_dir)

    speech_data = None
    if args.analyse_data:
        print('- loading dataset samples...')
        dataset_names = get_dataset_names(Path(args.input_path))
        if not dataset_names:
            dataset_names = dataset_ids
        dataset_samples = load_sampled_datasets(sampling_dir, dataset_names)
        assert dataset_samples, f'No dataset sample were loaded for analysis.'

        print('- analyzing datasets...')
        print(f'    - calculating sound duration statistics...')
        sound_duration_stat = calculate_duration_statistics(dataset_samples, tier_name=args.phone_tier)
        formant_data = dict()
        feature_data = dict()
        for ds_id in dataset_samples:
            print(f'    {ds_id}:')
            for f_id in dataset_samples[ds_id]:
                feature_data[ds_id][f_id] = {'file_name': f_id,
                                             'duration': dataset_samples[ds_id][f_id]['duration'],
                                             'analysis': {}
                                             }

            print(f'    - calculating speech rate...')
            feature_data[ds_id] = estimate_speech_rate(feature_data[ds_id], dataset_samples[ds_id], tier_name=args.phone_tier, stat_data=sound_duration_stat)

            print(f'    - calculating vowel space...')
            formant_data[ds_id] = estimate_vowel_space(dataset_samples[ds_id], tier_name=args.phone_tier)

            print(f'    - smoothing f0 values...')
            dataset_samples[ds_id] = process_melodic_data(dataset_samples[ds_id], tier_name=args.phone_tier)
            ds_stat_data = calculate_descriptive_statistics(dataset_samples[ds_id])

            print(f'    - calculating word-level f0 measures...')
            feature_data[ds_id] = estimate_word_melodic_features(feature_data[ds_id], dataset_samples[ds_id], ds_stat_data, main_tier_name=args.word_tier, sub_tier_name=args.phone_tier)

            print(f'    - calculating f0 measures...')
            feature_data[ds_id] = estimate_general_melodic_features(feature_data[ds_id], dataset_samples[ds_id], ds_stat_data)

        speech_data = {'formants': formant_data, 'features': feature_data}
        report = json.dumps(speech_data, ensure_ascii=False, indent=4, sort_keys=True)
        with open(feature_file, 'w', encoding='utf-8') as fo:
            fo.write(report)

    if not args.plot_figures and not args.statistical_tests:
        exit()
    # load speech data, if necessary
    print(f'loading features...')
    if speech_data is None and feature_file.is_file():
        with open(feature_file, 'r') as fi:
            speech_data = json.load(fi)
    feature_data = speech_data['features']
    formant_data = speech_data['formants']
    # Transpose data in terms of 'feature', 'dataset' factors
    transposed_data = dict()
    for ds in feature_data:
        for f_id in feature_data[ds]:
            for feature in feature_data[ds][f_id]['analysis']:
                if feature_data[ds][f_id]['analysis'][feature] is not None:
                    if feature not in transposed_data:
                        transposed_data[feature] = dict()
                    if ds not in transposed_data[feature]:
                        transposed_data[feature][ds] = []
                    transposed_data[feature][ds].append(feature_data[ds][f_id]['analysis'][feature])
    for feature in transposed_data:
        for ds in transposed_data[feature]:
            transposed_data[feature][ds] = np.array(transposed_data[feature][ds])
    feature_data = transposed_data

    # plot the measures
    if args.plot_figures:
        print(f'\nplotting figures...')
        assert feature_data, f'No features were loaded for plotting.'
        for feature in sorted(feature_data):
            if feature != 'vowel_duration':
                pass
            if feature not in analysis_setup:
                print(f'--- {feature}: the feature processing is not defined')
                continue
            print(f'--- {feature}')
            plot_unit = analysis_setup[feature]['units']
            plot_title = analysis_setup[feature]['plot_title']
            plotting_style = analysis_setup[feature]['plot_style']
            if plotting_style == 'line':
                data_to_plot = {k: np.mean(feature_data[feature][k]) for k in feature_data[feature]}
            else:
                data_to_plot = feature_data[feature]
            if plotting_style == 'line':
                plot_lines(data_to_plot, plot_title, figure_dir)
            elif plotting_style == 'boxplot':
                plot_boxplots(data_to_plot, plot_title, figure_dir)
            elif plotting_style == 'violinplot':
                plot_violin_plots(data_to_plot, plot_title, figure_dir)
            elif plotting_style == 'violinplot_paired':
                plot_paired_violin_plots(data_to_plot, plot_title, figure_dir, unit_label=plot_unit, to_compare=args.plot_comparison, mapping=datasets_to_compare)
        print(f'--- formants')
        plot_vowel_space(formant_data, figure_dir)

    if args.statistical_tests:
        print(f'\nstatistical analysis...')
        statistical_report = defaultdict(lambda: defaultdict(dict))
        for feature in sorted(feature_data):
            if feature not in analysis_setup:
                print(f'--- {feature}: it is not setup')
                continue
            ds_names = sorted(feature_data[feature])
            for i_ds in range(len(ds_names)):
                statistical_report[feature][ds_names[i_ds]][ds_names[i_ds]] = (0, 0)
                for j_ds in range(i_ds+1, len(ds_names)):
                    ds_distr1 = feature_data[feature][ds_names[i_ds]]
                    ds_distr2 = feature_data[feature][ds_names[j_ds]]
                    n = min(len(ds_distr1), len(ds_distr2))
                    ds_distr1 = np.random.choice(ds_distr1, n, replace=False)
                    ds_distr2 = np.random.choice(ds_distr2, n, replace=False)
                    normal1 = shapiro(ds_distr1).pvalue > 0.05
                    normal2 = shapiro(ds_distr2).pvalue > 0.05
                    if normal1 and normal2:
                        both_normal = 1
                        statistical_report[feature][ds_names[i_ds]][ds_names[j_ds]] = [both_normal] + list(ttest_ind(ds_distr1, ds_distr2))
                        statistical_report[feature][ds_names[j_ds]][ds_names[i_ds]] = [both_normal] + list(ttest_ind(ds_distr1, ds_distr2))
                    else:
                        both_normal = 0
                        statistical_report[feature][ds_names[i_ds]][ds_names[j_ds]] = [both_normal] + list(wilcoxon(ds_distr1, ds_distr2))
                        statistical_report[feature][ds_names[j_ds]][ds_names[i_ds]] = [both_normal] + list(wilcoxon(ds_distr1, ds_distr2))

        with open(statistical_report_file, 'w', encoding='utf-8') as fo:
            header = ['features', 'ds name', 'ds name', 'normality', 'statistics', 'p-value']
            header_line = '\t'.join(header) + '\n'
            fo.write(header_line)
            for f_id in sorted(statistical_report):
                for ds1_id in sorted(statistical_report[f_id]):
                    for ds2_id in sorted(statistical_report[f_id][ds1_id]):
                        values = [str(round(v, 3)) for v in statistical_report[f_id][ds1_id][ds2_id]]
                        line = '\t'.join([f_id, ds1_id, ds2_id] + values) + '\n'
                        fo.write(line)
    return None


def calculate_declination(values: np.ndarray, ref_function: Union[Callable, None] = None) -> Union[float, int, None]:
    if len(values) == 0:
        return None
    if args.relative_values:
        result = ref_function(values[-1], values[0])
    else:
        result = values[-1] - values[0]
    return result


def calculate_descriptive_statistics(dataset: dict[str, dict]) -> dict:
    ds_values = [dataset[f_id]['f0'] for f_id in sorted(dataset)]
    ds_values = np.concatenate(ds_values)
    ds_values = ds_values[ds_values != 0]
    ds_stat_data = {'mean': np.mean(ds_values),
                    'limits': [np.percentile(ds_values, args.outlier_percentile), np.percentile(ds_values, 100 - args.outlier_percentile)]
                    }
    return ds_stat_data


def calculate_duration_statistics(data: dict[str, dict], tier_name: str) -> dict[str, dict]:
    duration_data = dict()
    for ds_id in data:
        for f_id in data[ds_id]:
            for s in data[ds_id][f_id]['tiers'][tier_name]:
                if s.text not in duration_data:
                    duration_data[s.text] = dict()
                duration_data[s.text]['data'].append(s.duration())
    for s in duration_data:
        duration_data[s]['mean'] = np.mean(duration_data[s]['data'])
        duration_data[s]['std'] = np.std(duration_data[s]['data'])
        del duration_data[s]['data']
    return duration_data


def calculate_first_value(values: np.ndarray, ref_value: Union[int, float, None] = None, ref_function: Union[Callable, None] = None) -> Union[int, None]:
    if len(values) == 0:
        return None
    if args.relative_values and ref_value is not None:
        result = int(ref_function(values[0], ref_value))
    else:
        result = int(values[0])
    return result


def calculate_last_accent(values: np.ndarray, ref_function: Union[Callable, None] = None) -> Union[float, int, None]:
    if len(values) == 0:
        return None
    if len(values) == 1:
        return 0
    if args.relative_values and ref_function is not None:
        result = ref_function(values[-1], values[-2])
    else:
        result = values[-2] - values[-1]
    return result


def calculate_max(values: np.ndarray, ref_value: Union[int, float, None] = None, ref_function: Union[Callable, None] = None) -> Union[float, None]:
    if len(values) == 0:
        return None
    result = np.max(values)
    if args.relative_values and ref_value is not None:
        result = ref_function(result, ref_value)
    return result


def calculate_mean(values: np.ndarray, ref_value: Union[int, float, None] = None, ref_function: Union[Callable, None] = None) -> Union[float, None]:
    if len(values) == 0:
        return None
    if args.relative_values and ref_value is not None:
        result = ref_function(np.mean(values), ref_value)
    else:
        result = float(np.mean(values))
    return result


def calculate_min(values: np.ndarray, ref_value: Union[int, float, None] = None, ref_function: Union[Callable, None] = None) -> Union[float, None]:
    if len(values) == 0:
        return None
    result = np.min(values)
    if args.relative_values and ref_value is not None:
        result = ref_function(result, ref_value)
    return result


def calculate_positive_values_ratio(values: Union[list, np.ndarray, None]) -> Union[float, None]:
    if values is None:
        return None
    n_rises = sum(1 for v in values if v is not None and v > 0)
    rise_ratio = n_rises / len(values)
    return rise_ratio


def calculate_range(values: np.ndarray, ref_function: Union[Callable, None] = None) -> Union[float, int, None]:
    if len(values) == 0:
        return None
    if args.relative_values and ref_function is not None:
        result = ref_function(np.max(values), np.min(values))
    else:
        result = np.max(values) - np.min(values)
    return result


def calculate_std(values: np.ndarray, ref_value: Union[int, float, None] = None, ref_function: Union[Callable, None] = None) -> Union[float, None]:
    if len(values) == 0:
        return None
    if args.relative_values and ref_value is not None:
        result = np.std([abs(ref_function(v, ref_value)) for v in values])
    else:
        result = float(np.std(values))
    return result


def calculate_sound_durations(data: Union[IntervalTier, list[Interval]], norm_factors: Union[None, dict[str, dict]] = None) -> np.ndarray:
    if norm_factors is None:
        durations = [s.duration() for s in data]
    else:
        durations = [z_norm(s.duration(), norm_factors[s.text]['mean'], norm_factors[s.text]['std']) for s in data]
    durations = np.array(durations)
    return durations


def calculate_units_per_second(sounds: Union[IntervalTier, list[Interval]], mode: Union[None, str] = None) -> float:
    total_duration = sum(s.duration() for s in sounds)
    n_sounds = len(sounds)
    if mode == 'vowels':
        n_sounds = len([s for s in sounds if is_vowel(s.text)])
    result = None
    if n_sounds != 0:
        result = round(n_sounds/total_duration, 2)
    return result


def dataset_sampling(datasets: dict[str, dict[str, Union[float, np.ndarray]]], ds_names: Union[list[str], None] = None) -> dict[str, dict[str, Union[float, np.ndarray]]]:
    if ds_names is None:
        ds_names = sorted(datasets)
    sampling_datasets = {}
    for d in ds_names:
        print(f'  {d}, ({len(datasets[d])})')
        file_names = list(datasets[d].keys())
        random.shuffle(file_names)
        duration_threshold = args.dataset_sample_size * 3600  # hour2sec
        total_duration = 0.0
        target_files = []
        for f_id in file_names:
            target_files.append(f_id)
            total_duration += datasets[d][f_id]['duration']
            if duration_threshold <= total_duration:
                print(f'    {round(total_duration, 2)} sec.')
                break
        sampling_datasets[d] = {f: datasets[d][f] for f in target_files}
    return sampling_datasets


def estimate_general_melodic_features(feature_data: dict[str, dict], dataset: dict[str, dict], ds_stat_data: Union[None, dict] = None) -> dict[str, dict]:
    for f_id in sorted(dataset):
        lower_limit = ds_stat_data['limits'][0]
        upper_limit = ds_stat_data['limits'][1]
        f0_values = dataset[f_id]['f0'][(lower_limit < dataset[f_id]['f0']) & (dataset[f_id]['f0'] < upper_limit)]
        mean_value = calculate_mean(f0_values)
        feature_data[f_id]['analysis']['f0_mean'] = mean_value
        feature_data[f_id]['analysis']['f0_std'] = calculate_std(f0_values, mean_value, to_semitones)
        feature_data[f_id]['analysis']['f0_max'] = calculate_max(f0_values, mean_value, to_semitones)
        feature_data[f_id]['analysis']['f0_min'] = calculate_min(f0_values, mean_value, to_semitones)
        feature_data[f_id]['analysis']['f0_range'] = calculate_range(f0_values, to_semitones)
        rises, falls = get_rises_falls(f0_values)
        feature_data[f_id]['analysis']['f0_max_fall'] = get_max_range(falls)
        feature_data[f_id]['analysis']['f0_max_rise'] = get_max_range(rises)
        if args.relative_values and ds_stat_data is not None:
            feature_data[f_id]['analysis']['ds_f0_mean'] = ds_stat_data['mean']
            file_mean_value = to_semitones(feature_data[f_id]['analysis']['f0_mean'], ds_stat_data['mean'])
            feature_data[f_id]['analysis']['f0_mean'] = file_mean_value
    return feature_data


def estimate_word_melodic_features(feature_data: dict[str, dict], dataset: dict[str, dict], ds_stat_data: dict[str, dict]=None, main_tier_name: str ='words', sub_tier_name: str ='phones') -> dict[str, dict]:
    for f_id in sorted(dataset):
        # print(f_id)
        assert main_tier_name in dataset[f_id]['tiers'], f'{f_id}: There is no tier "{main_tier_name}" in the TextGrid data.'
        assert sub_tier_name in dataset[f_id]['tiers'], f'{f_id}: There is no tier "{sub_tier_name}" in the TextGrid data.'
        word_tier = [w for w in dataset[f_id]['tiers'][main_tier_name] if w.text not in {'', '<p:>'}]
        vowel_tier = [s for s in dataset[f_id]['tiers'][sub_tier_name] if is_vowel(s.text)]
        file_values = []
        for word in word_tier:
            word_values = []
            vowels = [v for v in vowel_tier if word.overlaps(v) and args.min_vowel_duration <= v.duration() <= args.max_sound_length]
            for vowel in vowels:
                start = int(vowel.start_time / args.feature_window_length)
                end = int(vowel.end_time / args.feature_window_length)
                vowel_values = dataset[f_id]['f0'][start:end+1]
                lower_limit = ds_stat_data['limits'][0]
                upper_limit = ds_stat_data['limits'][1]
                vowel_values = vowel_values[(lower_limit < vowel_values) & (vowel_values < upper_limit)]
                if len(vowel_values) != 0:
                    word_values.append(vowel_values)
            word_max_f0 = np.nan
            if len(word_values) > 0:
                word_max_f0 = np.max(np.concatenate(word_values))
            file_values.append(word_max_f0)
        file_values = np.array(file_values)
        file_values = file_values[~(np.isnan(file_values))]

        file_ref_value = None
        if len(file_values) > 0:
            file_ref_value = file_values[0]
        feature_data[f_id]['analysis']['first_word_f0'] = calculate_first_value(file_values)
        feature_data[f_id]['analysis']['last_word_accent'] = calculate_last_accent(file_values, to_semitones)
        feature_data[f_id]['analysis']['word_f0_mean'] = calculate_mean(file_values, file_ref_value, to_semitones)
        feature_data[f_id]['analysis']['word_f0_std'] = calculate_std(file_values, file_ref_value, to_semitones)
        feature_data[f_id]['analysis']['word_f0_max'] = calculate_max(file_values, file_ref_value, to_semitones)
        feature_data[f_id]['analysis']['word_f0_min'] = calculate_min(file_values, file_ref_value, to_semitones)
        feature_data[f_id]['analysis']['melody_declination'] = calculate_declination(file_values, to_semitones)
        feature_data[f_id]['analysis']['prosodic_words'] = len(file_values)
        feature_data[f_id]['analysis']['n_words'] = len(word_tier)
    accent_values = [feature_data[f_id]['analysis']['last_word_accent'] for f_id in dataset]
    accent_rise_ratio = calculate_positive_values_ratio(accent_values)
    for f_id in feature_data:
        feature_data[f_id]['analysis']['rise_accent_ratio'] = accent_rise_ratio
    return feature_data


def estimate_speech_rate(feature_data: dict[str, dict], dataset: dict[str, dict], tier_name: str, stat_data: Union[None, dict[str, dict]] = None) -> dict[str, dict]:
    for f_id in dataset:
        assert tier_name in dataset[f_id]['tiers'], f'{f_id}: There is no tier "{tier_name}" in the TextGrid data.'
        sounds = [s for s in dataset[f_id]['tiers'][tier_name] if s.text not in {'', '<p:>'}]
        vowels = [s for s in sounds if is_vowel(s.text)]

        feature_data[f_id]['analysis']['speech_rate'] = calculate_units_per_second(sounds, mode='vowels')
        durations = np.asarray(calculate_sound_durations(vowels, norm_factors=stat_data))
        feature_data[f_id]['analysis']['vowel_duration'] = calculate_mean(durations)
    return feature_data


def estimate_vowel_space(dataset: dict[str, dict], tier_name: str = 'phones') -> dict[str, dict]:
    formant_data = {'raw': {}, 'mean': {}, 'std': {}, 'sem': {}}
    for f_id in sorted(dataset):
        assert tier_name in dataset[f_id]['tiers'], f'{f_id}: There is no tier "{tier_name}" in the TextGrid data.'
        vowel_tier = select_vowels(dataset[f_id]['tiers'][tier_name])
        mid_timestamps = find_mid_sound_timestamps(vowel_tier, dataset[f_id]['timestamps'])
        for i, vowel in enumerate(vowel_tier):
            if mid_timestamps[i] is None:
                continue
            if is_voiced_timestamp(dataset[f_id]['timestamps'][mid_timestamps[i]], dataset[f_id]['voiced'], frame_length=0.025):
                if vowel.text not in formant_data['raw']:
                    formant_data['raw'][vowel.text] = {'f1': [], 'f2': []}
                f1_value = to_barks(dataset[f_id]['f1'][mid_timestamps[i]])
                f2_value = to_barks(dataset[f_id]['f2'][mid_timestamps[i]])
                if dataset[f_id]['f2'][mid_timestamps[i]] > 1500 and vowel.text[0] not in front_vowels:
                    pass
                    # print(f'{f_id}: {vowel}', dataset[f_id]['f2'][mid_timestamps[i]])
                formant_data['raw'][vowel.text]['f1'].append(f1_value)
                formant_data['raw'][vowel.text]['f2'].append(f2_value)
    for v_id in formant_data['raw']:
        formant_data['mean'][v_id] = {'f1': float(np.mean(formant_data['raw'][v_id]['f1'])),
                                      'f2': float(np.mean(formant_data['raw'][v_id]['f2']))
                                      }
        formant_data['std'][v_id] = {'f1': float(np.std(formant_data['raw'][v_id]['f1'])),
                                     'f2': float(np.std(formant_data['raw'][v_id]['f2']))
                                     }
        formant_data['sem'][v_id] = {'f1': float(sem(formant_data['raw'][v_id]['f1'])),
                                     'f2': float(sem(formant_data['raw'][v_id]['f2']))
                                     }
    del formant_data['raw']
    return formant_data


def find_mid_sound_timestamps(tier: list[Interval], timestamps: list[float]) -> list[int]:
    mid_point_idx = []
    for sound in tier:
        mid_estimation = (sound.start_time + sound.end_time) / 2
        i = np.searchsorted(timestamps, mid_estimation, side='left')
        if i == len(timestamps):
            mid_point_idx.append(None)
            continue
        # i < len(timestamps)-2: because the middle of the vowel shouldn't be the last formant measurement before voiceless signal.
        if i < len(timestamps)-2 and abs(timestamps[i] - mid_estimation) > abs(timestamps[i+1] - mid_estimation):
            i += 1
        if not timestamps[i] in sound:
            i = None
        mid_point_idx.append(i)
    return mid_point_idx


def get_dataset_names(path: Path) -> list[str]:
    dataset_paths = path.rglob('*.tar')
    dataset_paths = [p for p in dataset_paths if p.with_suffix('.f0').is_file()]
    ds_names = [d.stem for d in dataset_paths]
    return ds_names


def get_max_range(arr: Union[list, np.ndarray]) -> Union[None, float, int]:
    max_range = None
    for i, elem in enumerate(arr):
        if len(elem) == 0:
            continue
        current_range = calculate_range(elem, to_semitones)
        if max_range is None or max_range < current_range:
            max_range = current_range
    return max_range


def get_ipu_tier(tier: IntervalTier) -> list[Interval]:
    ipu_tier = []
    for interval in tier:
        if len(ipu_tier) == 0 or interval.text == '' or ipu_tier[-1].text == '':
            ipu_tier.append(interval)
        else:
            ipu_tier[-1] = ipu_tier[-1] + interval
    return ipu_tier


def get_rises_falls(arr: Union[list, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if len(arr) < 2:
        return np.array([]), np.array([])
    rises = []
    falls = []
    current_rise = []
    current_fall = []
    current_trend = 0
    i = 1
    while i < len(arr):
        if arr[i] > arr[i - 1]:
            current_rise.append(arr[i-1])
            current_trend = 1
            if len(current_fall) > 0:
                current_fall.append(arr[i-1])
                falls.append(copy.deepcopy(current_fall))
                current_fall = []
        elif arr[i] < arr[i - 1]:
            current_fall.append(arr[i-1])
            current_trend = -1
            if len(current_rise) > 0:
                current_rise.append(arr[i-1])
                rises.append(copy.deepcopy(current_rise))
                current_rise = []
        else:
            if current_trend == 1:
                current_rise.append(arr[i-1])
            else:
                current_fall.append(arr[i-1])
        i += 1

    # assign the last value to the final rise or final fall
    if arr[-2] < arr[-1]:
        current_rise.append(arr[-1])
    elif arr[-2] > arr[-1]:
        current_fall.append(arr[-1])
    elif arr[-2] == arr[-1] and len(current_rise) > 0:
        current_rise.append(arr[-1])
    elif arr[-2] == arr[-1] and len(current_fall) > 0:
        current_fall.append(arr[-1])

    # put the final melodic movement to the list of rises or falls
    if len(current_rise) > 0:
        rises.append(copy.deepcopy(current_rise))
    elif len(current_fall) > 0:
        falls.append(copy.deepcopy(current_fall))
    rises = np.array(rises)
    falls = np.array(falls)
    return rises, falls


def is_voiced_timestamp(timestamp: float, voicing_data: list[bool], frame_length: float = 0.025) -> bool:
    voicing_points = np.arange(0, len(voicing_data)*frame_length, frame_length)
    i = np.searchsorted(voicing_points, timestamp)
    return voicing_data[i]


def is_voiceless_sound(label: str) -> bool:
    if label == '':
        return True
    for symbol in voiceless_consonant_symbols:
        if symbol in label:
            return True
    return False


def is_vowel(label: str) -> bool:
    for symbol in vowel_symbols:
        if symbol in label:
            return True
    return False


def load_features(path: Path, data: Union[None, dict] = None, feature_label: str = '') -> dict[str, dict[str, Union[float, int]]]:
    if data is None:
        data = dict()
    with open(path, 'rb') as fin:
        feature_data = pickle.load(fin)
    for f_id in sorted(feature_data['data']):
        if f_id not in data:
            data[f_id] = {}
        if type(feature_data['data'][f_id]) is float:
            data[f_id][feature_label] = feature_data['data'][f_id]
        else:
            for f_tier in feature_data['data'][f_id]:
                data[f_id][f_tier] = feature_data['data'][f_id][f_tier]
    return data


def load_raw_data(path: Path, ds_names: Union[None, list[str]] = None, reload_datasets: bool = False) -> dict[str, dict]:
    ds_file = Path(args.output_path, 'datasets.pkl')
    if reload_datasets or not ds_file.is_file():
        dataset_paths = path.rglob('*.f0')
        dataset_paths = [p for p in dataset_paths if p.with_suffix('.duration').is_file()]
        dataset_paths = [p for p in dataset_paths if p.with_suffix('.formant').is_file()]
        dataset_paths = [p for p in dataset_paths if p.with_suffix('.aligned').is_dir() or p.with_suffix('.aligned.tar').is_file()]
        if ds_names is not None:
            dataset_paths = [p for p in dataset_paths if p.stem in ds_names]
        datasets = defaultdict(dict)
        for ds_path in sorted(dataset_paths):
            ds_id = ds_path.stem
            print(ds_id)
            datasets[ds_id] = load_features(ds_path.with_suffix('.duration'), datasets[ds_id], 'duration')
            datasets[ds_id] = load_features(ds_path.with_suffix('.f0'), datasets[ds_id])
            datasets[ds_id] = load_features(ds_path.with_suffix('.formant'), datasets[ds_id])
            datasets[ds_id] = load_segmentation(ds_path.with_suffix('.aligned'), datasets[ds_id])
            files_with_no_alignment = [f for f in datasets[ds_id] if 'tiers' not in datasets[ds_id][f]]
            for f in files_with_no_alignment:
                del datasets[ds_id][f]
        with open(ds_file, 'wb') as fin:
            pickle.dump(datasets, fin)
    else:
        with open(ds_file, 'rb') as fin:
            datasets = pickle.load(fin)
    return datasets


def load_sampled_datasets(path: Path, ds_names=None) -> dict[str, dict]:
    if ds_names is not None:
        files = sorted(path.glob('*.pkl'))
        file_names = [f.stem for f in files]
        ds_to_sample = [d for d in ds_names if d not in file_names]
        if len(ds_to_sample) != 0:
            dataset_data = load_raw_data(Path(args.input_path), ds_names=ds_to_sample, reload_datasets=True)
            dataset_samples = dataset_sampling(dataset_data, ds_to_sample)
            save_dataset_samples(dataset_samples, path)
    files = sorted(path.glob('*.pkl'))
    assert len(files) != 0, f'No files to load. Datasets were not loaded: {ds_names}.'
    dataset = {}
    for file in sorted(files):
        if ds_names is not None and file.stem not in ds_names:
            continue
        with open(file, 'rb') as fi:
            dataset[file.stem] = pickle.load(fi)
    return dataset


def load_segmentation(input_path: Path, data: Union[None, dict[str, dict]] = None) -> dict[str, dict]:
    print(f'loading segmentation...')
    remove_path = False
    if data is None:
        data = defaultdict(dict)
    path = input_path
    if not input_path.is_dir() and input_path.with_suffix('.aligned.tar').is_file():
        path = input_path.with_suffix('.aligned_tmp')
        remove_path = True
        with tarfile.open(input_path.with_suffix('.aligned.tar'), 'r') as tar:
            tar.extractall(path=path)
    for root, dirs, files in sorted(os.walk(path)):
        files = [Path(root, file) for file in files if file.endswith('.TextGrid')]
        for file in tqdm(sorted(files)):
            tg = TextGrid.from_file(file)
            data[file.stem]['tiers'] = {tier.name: tier for tier in tg.objects}
    if remove_path:
        shutil.rmtree(path)
    return data


def plot_boxplots(data: dict[str, Union[np.ndarray, list]], label: str, figure_path: Path):
    labels = sorted(data)
    plt.boxplot([data[k] for k in labels], labels=[_clear_x_tick_label(k) for k in labels], showfliers=False)
    plt.title(f'{label}', fontweight='bold')
    plt.xticks(rotation=90)
    plt.xlabel('datasets')
    plt.grid(True)
    plt.tight_layout(pad=1.0)
    if args.show_figures:
        plt.show()
    figure_file = Path(figure_path, f'{label}.png')
    plt.savefig(figure_file, bbox_inches='tight')
    if args.plot_pdf:
        plt.savefig(figure_file.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


def plot_lines(data: dict[str, Union[np.ndarray, list]], label: str, figure_path: Path):
    x_labels = sorted(list(data.keys()))
    x_ticks = list(range(len(x_labels)))
    values = [data[k] if k in data else None for k in x_labels]
    sems = [sem(data[k], nan_policy='omit') if k in data else None for k in x_labels]
    plt.plot(values, linewidth=2, color='black')
    alpha_value = 0.1
    if args.paper_plots:
        alpha_value = 0.05
    if args.show_sem:
        plt.fill_between(x=list(range(len(values))),
                         y1=[values[i] - sems[i] if values[i] is not None else None for i in range(len(values))],
                         y2=[values[i] + sems[i] if values[i] is not None else None for i in range(len(values))],
                         color='black', alpha=alpha_value)
    if not args.paper_plots:
        plt.title(f'{label}', fontweight='bold')
    plt.xticks(x_ticks, x_labels, rotation='vertical', fontsize=args.font_size)
    plt.yticks(fontsize=args.font_size)
    plt.xlabel('age (months)', fontsize=args.font_size)
    plt.tight_layout(pad=1.0)
    if args.show_figures:
        plt.show()
    figure_file = Path(figure_path, f'{label}.png')
    plt.savefig(figure_file, bbox_inches='tight')
    if args.plot_pdf:
        plt.savefig(figure_file.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


def plot_paired_violin_plots(data: dict[str, Union[np.ndarray, list]], label: str, figure_path: Path, unit_label: str = '', to_compare: str = 'style', mapping: [None, list[str]] = None):
    y_label = label
    if unit_label != '':
        y_label = unit_label
    paired_df = [item for sublist in mapping for item in sublist if len(sublist) == 2]
    single_df = [item for sublist in mapping for item in sublist if len(sublist) == 1]
    paired_df_to_plot = to_dataframe({k: data[k] for k in paired_df if 'original' not in k}, y_label)
    single_df_to_plot = to_dataframe({k: data[k] for k in data if 'original' in k}, y_label)
    if label == 'Number of vowels per second':
        paired_df_to_plot = paired_df_to_plot[paired_df_to_plot[y_label] < 8]
    elif label == 'Vowel duration':
        paired_df_to_plot = paired_df_to_plot[paired_df_to_plot[y_label] < 2]
    two_colours = [sns.color_palette(palette='RdBu')[-1], sns.color_palette(palette='RdBu')[0]]
    if args.paper_plots:
        two_colours = [(0, 0.4470, 0.7410), (0.8500, 0.3250, 0.0980)]
    single_colours = [sns.color_palette(palette='RdBu')[1]]
    single_df = [s.replace('ljs.original', 'original') for s in single_df]
    if to_compare == 'style':
        sns.violinplot(data=paired_df_to_plot, x='content', y=y_label, hue='style', palette=two_colours, split=True, inner="quart")
        if single_df:
            sns.violinplot(data=single_df_to_plot, x='content', y=y_label, hue='style', palette=single_colours, legend=single_df, inner="quart")
    else:
        sns.violinplot(data=paired_df_to_plot, x='style', y=y_label, hue='content',  palette=two_colours, split=True, inner="quart")
        if single_df:
            sns.violinplot(data=single_df_to_plot, x='style', y=y_label, hue='content', palette=single_colours, legend=single_df, inner="quart")
    # if args.paper_plots:
    #     sns.set(font_scale=1)
    plt.grid(axis='y')
    if not args.paper_plots:
        plt.title(f'{label}', fontweight='bold')
    if args.show_figures:
        plt.show()

    figure_file = Path(figure_path, f'{label}.png')
    plt.savefig(figure_file, bbox_inches='tight')
    if args.plot_pdf:
        plt.savefig(figure_file.with_suffix('.pdf'), bbox_inches='tight')
        plt.savefig(figure_file.with_suffix('.svg'), bbox_inches='tight')
    plt.close()


def plot_violin_plots(data: dict[str, Union[np.ndarray, list]], label: str, figure_path: Path):
    cutoff = 20
    labels = sorted(data)
    if args.violin_plot_with_quartiles:
        data = copy.deepcopy(data)
        for k in labels:
            limits = np.percentile(data[k], [cutoff, 100-cutoff])
            data[k] = data[k][(data[k] >= limits[0]) & (data[k] <= limits[1])]
    plt.violinplot([data[k] for k in labels], showmeans=True)
    plt.title(f'{label}', fontweight='bold')
    x_ticks = [x+1 for x in range(len(labels))]
    x_labels = [k for k in labels]
    plt.xticks(x_ticks, x_labels, rotation='vertical')
    plt.xlabel('datasets')
    plt.grid(True)
    if args.show_figures:
        plt.show()
    figure_file = Path(figure_path, f'{label}.png')
    plt.savefig(figure_file, bbox_inches='tight')
    if args.plot_pdf:
        plt.savefig(figure_file.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


def plot_vowel_space(data: dict[str, dict], figure_path: Path):
    in_hertz = True
    min_radius = 0.0
    ds_labels = sorted(data)
    fig, axs = plt.subplots(3, math.ceil(len(ds_labels)/3), figsize=(12, 6))

    axs = axs.flatten()
    for i, ds_id in enumerate(ds_labels):
        ax = axs[i]
        # correct_vowels = [v for v in correct_vowels if v.endswith('ː')]
        correct_vowels = [v for v in sorted(data[ds_id]['mean']) if len(v) == 1 or v.endswith('ː')]
        # print(correct_vowels)
        for v in correct_vowels:
            center = (data[ds_id]['mean'][v]['f2'], data[ds_id]['mean'][v]['f1'])
            y = data[ds_id]['std'][v]['f1'] + min_radius
            x = data[ds_id]['std'][v]['f2'] + min_radius
            if in_hertz:
                center = bark2hertz(center)
                y = bark2hertz(y)
                x = bark2hertz(x)
            ax.annotate(f'{v}', center, color='black', ha='center', va='center')
            ellipse = patches.Ellipse(center, width=x, height=y, edgecolor='black', facecolor='none', label=v)
            # ax.add_patch(ellipse)
        ylims = [3, 9]
        xlims = [7, 16]
        if in_hertz:
            ylims = bark2hertz(ylims)
            xlims = bark2hertz(xlims)
        ax.set_ylim(ylims)
        ax.set_xlim(xlims)
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{ds_id}')
        ax.set_ylabel('F1 (Barks)')
        ax.set_xlabel('F2 (Barks)')
        ax.grid(True)
    if len(axs) > len(ds_labels):
        diff = len(axs) - len(ds_labels)
        for i in range(1, diff+1):
            fig.delaxes(axs[-i])
    fig.suptitle('Vowel space', fontsize=16)
    if args.show_figures:
        plt.show()
    figure_file = Path(figure_path, f'Vowels space.png')
    if in_hertz:
        figure_file = figure_file.with_stem('Vowels space (in Hz)')
    plt.savefig(figure_file, bbox_inches='tight')
    if args.plot_pdf or in_hertz:
        plt.savefig(figure_file.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


# 1. split speech signal IPU/pauses based on MFA segmentation.
# 2. pauses: assign 0 to all F0 values.
# 3. IPUs: interpolate voiceless consonants
# 4. smooth all F0 values.
def process_melodic_data(dataset: dict[str, dict], tier_name: str = 'phones') -> dict[str, dict]:
    for f_id in sorted(dataset):
        assert tier_name in dataset[f_id]['tiers'], f'{f_id}: There is no tier "{tier_name}" in the TextGrid data.'
        raw_values = copy.deepcopy(dataset[f_id]['f0'])
        dataset[f_id]['f0_raw'] = dataset[f_id]['f0']

        tier_data = dataset[f_id]['tiers'][tier_name]
        voiceless_sounds = [v for v in tier_data if is_voiceless_sound(v.text)]
        for sound in voiceless_sounds:
            start = int(sound.start_time / args.feature_window_length)
            end = int(sound.end_time / args.feature_window_length)
            raw_values[start:end+1] = 0

        ipu_tier = get_ipu_tier(dataset[f_id]['tiers'][tier_name])
        for ipu in ipu_tier:
            if ipu.text == '':
                continue
            start = int(ipu.start_time / args.feature_window_length)
            end = int(ipu.end_time / args.feature_window_length)
            raw_values[start:end+1] = smooth_melodic_data(raw_values[start:end+1])
        dataset[f_id]['f0'] = raw_values
    return dataset


def save_dataset_samples(data: dict[str, dict], output_dir: Path):
    for ds_id in data:
        bin_path = Path(output_dir, f"{ds_id}.pkl")
        with open(bin_path, 'wb') as fo:
            pickle.dump(data[ds_id], fo)
    return None


def select_vowels(tier: IntervalTier) -> list[Interval]:
    vowels = []
    for i, sound in enumerate(tier):
        if not is_vowel(sound.text):
            continue
        if i > 0 and sound.text[0] not in front_vowels and (tier[i-1].text.endswith('ʲ') or tier[i-1].text.endswith('j')):
            pass
            # continue
        if len(sound.text) > 1 and not sound.text.endswith(':'):
            continue
        if sound.duration() < 0.1:
            continue
        vowels.append(sound)
    return vowels


# (1) interpolate zero-values,
# (2) smooth,
# (3) interpolate previously-zeroed-valued once again as they could lose interpolation as a result of smoothing.
def smooth_melodic_data(melodic_data: np.ndarray) -> np.ndarray:
    non_zero_indices = np.nonzero(melodic_data)[0]
    if len(non_zero_indices) == 0:
        return melodic_data
    zeros_prefix = np.linspace(0, non_zero_indices[0], num=non_zero_indices[0]+1, dtype=int)[:-1]
    zeros_suffix = np.linspace(non_zero_indices[-1], len(melodic_data)-1, num=len(melodic_data) - non_zero_indices[-1], dtype=int)[1:]
    zero_indices = np.concatenate((zeros_prefix, zeros_suffix))
    interpolated_values = np.interp(np.arange(len(melodic_data)), non_zero_indices, melodic_data[non_zero_indices])
    smoothed_values = interpolated_values
    start_idx = len(zeros_prefix)
    end_idx = len(smoothed_values) - len(zeros_suffix)
    if end_idx - start_idx >= args.f0_smoothing_window:
        smoothed_values[start_idx:end_idx] = savgol_filter(smoothed_values[start_idx:end_idx], window_length=args.f0_smoothing_window, polyorder=3)
    smoothed_values = np.interp(np.arange(len(melodic_data)), non_zero_indices, smoothed_values[non_zero_indices])
    smoothed_values[zero_indices] = 0
    return smoothed_values


def to_barks(hz_value: Union[int, float, np.ndarray, None]) -> Union[float, np.ndarray, None]:
    if hz_value is None:
        return None
    result = 13 * np.arctan(0.00076 * hz_value) + 3.5 * np.arctan((hz_value / 7500) ** 2)
    return result


def to_dataframe(input_data: dict[str, Union[float, int, Iterable]], feature_label: str) -> pd.DataFrame:
    resulting_data = []
    for ds_id in input_data:
        ds_speech = 'original'
        if 'cds' in ds_id:
            ds_speech = 'cds'
        elif 'neutral' in ds_id:
            ds_speech = 'neutral'
        elif 'read' in ds_id:
            ds_speech = 'read'
        elif 'conv' in ds_id:
            ds_speech = 'conversation'
        ds_content = 'ljs.original'
        if 'librispeech' in ds_id:
            ds_content = 'librispeech'
        elif 'childes' in ds_id:
            ds_content = 'childes'
        elif 'giles' in ds_id:
            ds_content = 'giles'
        for v in input_data[ds_id]:
            unit = {feature_label: v, 'style': ds_speech, 'content': ds_content}
            resulting_data.append(unit)
    resulting_df = pd.DataFrame(resulting_data)
    return resulting_df


def to_decibels(first: Union[int, float, None], second: Union[int, float, None]) -> Union[float, None]:
    if first is None or first == 0:
        return None
    if second is None or second == 0:
        return None
    value = 10 * math.log10(first/second)
    return value


def to_semitones(first: Union[int, float, None], second: Union[int, float, None]) -> Union[float, None]:
    if first is None or first == 0:
        return None
    if second is None or second == 0:
        return None
    value = 12 * math.log2(first/second)
    return value


def bark2hertz(bark: Union[int, float, Iterable]) -> Union[float, np.ndarray]:
    bark = np.array(bark)
    return 600 * np.sinh(bark / 6)


def z_norm(data: Union[int, float, Iterable], mean: Union[float, None] = None, std: Union[float, None] = None) -> Union[float, np.ndarray]:
    if type(data) is float or type(data) is int:
        return (data - mean) / std
    data = np.asarray(data)
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    z_norm_data = (data - mean) / std
    return z_norm_data


def _clear_x_tick_label(text: str) -> str:
    text = os.path.split(text)[-1].replace('_good', '').replace('GILES_', '').replace('neutral', 'neu').replace('librispeech', 'LS')
    return text


if __name__ == '__main__':
    main()
