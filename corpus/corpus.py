"""
This file contains all functions to read the corpus.

@author: Masoumeh Moradipour Tari
@date: July 2019
@place: Germany, University of Tuebingen
"""

import os
import pandas as pd
import numpy as np
from corpus.audio import get_sig_of_words, read_wav_file, get_boundary_of_words, split_signal, get_summary_freq_band
from Exception.exception import save_as_exception
from pathlib import Path
from util.utility import read_rds_file


def read_corpus_file(words_df, file, audio_basedir, include_wav, sample_rate):
    """
    Reading one wav file in the corpus.

    Args:
        words_df : A data frame contains all information belong to the specific audio file.
        file (str): The audio file in the Corpus.
        audio_basedir (str) : The path to the corpus.
        include_wav: The data frame contains the wav form as well or not.
        sample_rate (int) : Sample rate.

    Returns:
        pandas.DataFrame with columns wordtoken, start, end, File, Prev, triphones, FBSFs
    """
    try:
        wav_dir = os.path.join(audio_basedir, file)
        print("Reading the file: ", wav_dir)
        wav = read_wav_file(wav_dir, sample_rate)
        word_signals = get_sig_of_words(wav, words_df.start.values, words_df.end.values, sample_rate)
        word_boundaries = get_boundary_of_words(word_signals, np.argmin, 800, 1000)
        word_boundary_parts = split_signal(word_signals, word_boundaries)
        rds_df = words_df.drop(['FBSFs'], axis=1)
        rds_df["FBSFs"] = get_summary_freq_band(word_boundary_parts, sample_rate)
        rds_df["chunk_index"] = word_boundaries
        rds_df["chunk_duration"] = rds_df["chunk_index"].apply(lambda x: x / sample_rate)
        if include_wav:
            rds_df["wav"] = word_signals
        return rds_df
    except Exception as e:
        save_as_exception(audio_basedir, file, e)


def read_corpus_files(rds_df, audio_basedir, include_wav, sample_rate):
    """
    Reading all files in the corpus.

    Args:
        rds_df : A data frame contains all words and wav file path
        audio_basedir (str) : dir path to audio files
        include_wav: The data frame contains the wav form as well or not.
        sample_rate (int) : Sample rate.

    Returns:
        pandas.DataFrame with columns wordtoken, start, end, File, Prev, triphones, FBSFs
    """

    files = rds_df.File.unique().tolist()
    df_all_files = []
    for file in files:
        try:
            if Path(os.path.join(audio_basedir, file)).exists():
                df_all_files.append(read_corpus_file(rds_df.loc[rds_df['File'] == file], file, audio_basedir, include_wav, sample_rate))
        except Exception as e:
            save_as_exception(audio_basedir, file, e)
            print(e)
            pass

    return df_all_files


def read_corpus(rds_file_path, audio_basedir, include_wav, sample_rate):
    """
    Reading all files in the corpus.

    Args:
        rds_file_path (str): dir path to rds file
        audio_basedir (str) : dir path to audio files
        include_wav: The data frame contains the wav form as well or not.
        sample_rate (int) : Sample rate.

    Returns:
        pandas.DataFrame with columns wordtoken, start, end, File, Prev, triphones, FBSFs
    """
    try:
        # get the word list from rds file
        words_df = read_rds_file(rds_file_path)
        return pd.concat(read_corpus_files(words_df, audio_basedir, include_wav, sample_rate), ignore_index=True)
    except Exception as e:
        print(e)




