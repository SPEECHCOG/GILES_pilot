### Code for Räsänen & Kocharov (2024). "A Pipeline for Stochastic and Controlled Generation of Realistic Language Input for Simulating Infant Language Acquisition".

Program code for:
- Training a GPT-2 architecture language model (LM) from CHILDES transcripts and generating new child-directed speech transcripts using the model.
- Linguistic analysis of the generated and original transcripts.
- Prosodic analysis and comparison of the generated speech datasets.
- Visualization of the language learning simulation.

### 1. Contents:
1.1. Text generation and analysis: collecting training data, model training, data generation, data analysis
- `GILES_main.py`: a script to train a model and generate text using the model.
- `compare_datasets.py`: a script to linguistically analyze of text samples. The script generates csv-files with measure values for the given collections of texts.
- `measure_unique_utterances.py`: a script to analyze the proportion of unique utterances in the generated texts that never occur in CHILDES (or counting unique utterances in CHILDES).
- `analyze_results.m`: a script to plot the results based on the corpus analyses (MATLAB).
- `get_childes_naenglish.R`: the R script to download NA English corpora.
- `childes_to_ao_dataset.py`: the script to extract of CHILDES transcript data from downloaded CSV-files into age-specific bins.
- `analysis_functions.py`, `analysis_parameters_text.json`: contains functions and global parameters for linguistic analysis of the texts (used in `compare_datasets.py`).
- `cellfind.m`, `drawstds.m`, `ES_from_ttest.m`, `teekuva.m`, `Violin.m`, `violinplot.m`: contain functions used in `analyze_results.m`.

1.2. Speech data analysis: feature extraction, prosodic analysis of speech data
- `calculate_durations.py`: a script to calculate durations of speech sounds using the alignment produced by `calculate_durations.py`.
- `calculate_f0.py`: a script to calculate f0 values using pYIN algorithm (`librosa.pyin` implementation).
- `calculate_formants.py`: a script to calculate formants using Praat.
- `run_mfa_alignment.py`: a script to align speech and transcripts by means of Montreal Forced Aligner.
- `prosodic_analysis.py`: a script to linguistically analyze of speech samples using prosodic features extracted by `calculate_durations.py`, `calculate_f0.py`, `calculate_formants.py`. 
- `annotation_utils.py`: contains classes to handle speech segmentation in TextGrid (Praat) format.
- `analysis_parameters_speech.json`, `dataset_names.json`, `dataset_paths.json`, `dataset_alignments.json`: contains global parameters and a list of datasets for prosodic analysis of the speech.

1.3. Language learning simulations
- `eval_abx_checkpoints.m`: a script to convert ABX results in a required format (MATLAB).
- `eval_lextest_checkpoints.m`: a script to run LexTest and convert results in a required format (MATLAB).
- `plot_results.m`: a script to plot the ABX and LexTest results comparison (MATLAB).

### 2. Main dependencies
#### 2.1. Python

2.1.1. For LM training:
- tensorflow==2.12.1
- tensorflow-text==2.12.0

2.1.2. For evaluation of the generated text data:
- evaluate (https://huggingface.co/docs/evaluate/en/installation)
- stanza (https://stanfordnlp.github.io/stanza/)
- matplotlib
- pandas
- scipy

2.1.3. For evaluation of the generated speech data:
- librosa (https://github.com/librosa/librosa)
- Montreal Forced Aligner (https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)
  - Download latest acoustic model `english_us_mfa` and pronunciation dictionary `english_mfa` as it is explained in the MFA documentation.
- parselmouth (https://github.com/YannickJadoul/Parselmouth)
- matplotlib
- pandas
- scipy
- seaborn

2.1.4. Visualization of the language learning simulations:
- CDI_lextest (https://github.com/SPEECHCOG/CDI_lextest)
- CPC - contrastive predictive coding (https://github.com/MarvinLvn/CPC2)


#### 2.2. R
- childesr

### 3. Instructions:
#### 3.1. Text generation and analysis.
3.1.1. Download CHILDES transcripts from NA English corpora using `get_childes_naenglish.R` as CSV-files (see the list of corpora in the script). The script uses `childesr` library (https://github.com/langcog/childesr). Setup a path to directory, where you want to store CSV-files.
3.1.2. Extract CHILDES transcript data from downloaded CSV-files into age-specific bins using `childes_to_ao_dataset.py`
```
childes_to_ao_dataset.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR
```
3.1.3. Run `GILES_main.py` to train the model and generate transcripts with it (after setting data paths inside the file).
3.1.4. Run `compare_datasets.py` to extract linguistic features for the given texts (or collections of texts):
```
compare_datasets.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR --sample-data --analyze-data
```
3.1.5. The same code give a possibility to check statistical significance of feature changes along with (1) an age of a target child and (2) a number of words in a collection of texts corresponding to an age bin.
```
compare_datasets.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR --sample-data --analyze-data --statistical_tests
```
3.1.6. Run `analyze_results.m` to plot the feature comparison of the generated transcripts against original transcripts (set input and output paths inside the file).


#### 3.2. Speech analysis.
The current repository doesn't contain the code of TTS model. It contains only the code for prosodic analysis and 
comparison of the generated speech samples (datasets). The speech datasets should be stored as either separate 
directories with wav-files or uncompressed TAR files. Change the list of datasets to be processed in each of 
the following files: `dataset_names.json`, `dataset_paths.json`, `dataset_alignments.json`.

The speech data could be generated by any TTS. The authors used FastPitch model in their experiments (https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch)

3.2.1. Align speech and text transcripts using MFA
```
run_mfa_alignment.py --data-dir DATA_DIR --datasets dataset_alignments.json
```

3.2.2. Calculate file durations
```
calculate_durations.py --data-dir DATA_DIR --datasets dataset_paths.json
```

3.2.3. Calculate f0
```
calculate_f0.py --data-dir DATA_DIR --datasets dataset_paths.json
```

3.2.4. Calculate formants
```
calculate_formants.py --data-dir DATA_DIR --datasets dataset_paths.json
```

3.2.5. Caclculate and compare prosodic features.
```
prosodic_analysis.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR --datasets dataset_names.json
```


#### 3.2. Language learning simulation.
The language learning simulations were done using Contrastive Predictive Coding (CPC) implemented by M. Lavechin and collegues: https://github.com/MarvinLvn/CPC2 (M. Lavechin et al., Modeling early phonetic acquisition from child-centered audio data. Cognition, 245, 105734).

The experiments were run using following command:
```
python train.py --pathCheckpoint  $MODEL_DIR --pathDB $DATA_DIR --file_extension .wav --nLevelsGRU 2 --multihead_rnn --nEpoch 100 --random_seed 42 --n_process_loader 1 --save_step 2 --max_size_loaded 200000000  --schedulerRamp 10 --dropout --samplingType=samesequence --nGPU 1
```
The following MATLAB code was used to visualize the results of the language learning. Note, one should set up all path variables within the code. `eval_lextest_checkpoints.m` requires CDI_lextest (https://github.com/SPEECHCOG/CDI_lextest) to be installed.
```
eval_abx_checkpoints.m
eval_lextest_checkpoints.m
plot_results.m
```















