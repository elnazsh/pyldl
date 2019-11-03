"""
This file contains all functions to read the audio files and compute Frequency Band Summary.

@author: Masoumeh Moradipour Tari
@date: August 2019
@place: Germany, University of Tuebingen
"""

import librosa
from scipy import signal
import numpy as np
import python_speech_features
from Exception.exception import save_as_exception


def get_boundary_of_words(word_signals, function, smooth, window):
    """
       Get boundaries of all words in the corpus using a defined function.

        Args:
            word_signals (numpy array): wav object has been read by read_wav_file()
            function : A function which will be apply on enveloped word. For example: np.argmin.
            smooth : Smoothing degree, here is 800.
            window (int) : The sample long window (For example: 1000)
       Returns:
          A list of all word boundaries in the Corpus.
    """
    boundaries = []
    for sig in word_signals:
        boundaries.append(get_boundary(sig, function, smooth=smooth, window=window))
    return boundaries


def get_sig_of_words(wav, start, stop, sample_rate):
    """
       Get of a words in the corpus using start and stop indices.

        Args:
           wav (numpy array): wav object has been read by read_wav_file()
           start (real): start time in milli second.
           stop (real): end time in milli second.
           sample_rate (int) : Sample rate.

       Returns:
          A list of word signals. (list of lists)
    """
    word_wav_indices = zip((start * sample_rate).astype(int), (stop * sample_rate).astype(int))
    return [wav[start:stop] for start, stop in word_wav_indices]


def read_wav_file(file, sample_rate):
    """
       Read a wav file using librosa library.
        Args:
           file (string) : A wav file.
           sample_rate (int) : Sample rate.
       Returns:
          A sample rate and an array (digital form of a signal).
    """
    wav, rate = librosa.load(file, sr=44100)
    wav = librosa.resample(wav, rate, sample_rate)
    return wav


def envelope(sig, smooth):
    """
    Compute the analytic signal, using the Hilbert transform. (Amplitude envelope)
     Args:
        sig : Signal data. Must be real.
        smooth : smoothing degree.
    Returns:
        Analytic signal of x, of each 1-D array along axis.
    """
    try:
        analytic_signal = signal.hilbert(sig)
        amplitude_env = np.absolute(analytic_signal)
        if 0 < smooth < len(amplitude_env):
            smoothing_win = signal.windows.boxcar(smooth) / smooth
            smooth_env = np.convolve(amplitude_env, smoothing_win, mode='same')
            return smooth_env
        else:
            return amplitude_env
    except Exception as e:
        save_as_exception("Root", "Envelope", e)


def get_mel_spec(sig, sample_rate):
    """
     Compute log Mel-filter bank energy features from an audio signal.

      Args:
         sig (array like) : Signal data. Must be real.
         sample_rate (int) : Sample rate.
     Return:
         Mel spectrum.
     """
    try:
        mel_spec = python_speech_features.logfbank(sig, samplerate=sample_rate, winlen=0.005, winstep=0.005, nfilt=21,
                                                   preemph=0.97)
        mel_spec = mel_spec.T
        mini = np.amin(mel_spec)
        maxi = np.amax(mel_spec)
        if mini == maxi:
            return None
        else:
            mel_spec = np.ceil((mel_spec - mini) * (5 / np.abs(mini - maxi)))
            return mel_spec
    except Exception as e:
        save_as_exception("Root", "Mel Spectrum", e)
        return None


def get_boundary(word_sig, function, smooth, window):
    """
    Find the boundaries based on the defined function.

     Args:
        word_sig : An array of digital signal for each word.
        function : A function which will be apply on enveloped word. For example: np.argmin.
        smooth : Smoothing degree, here is 800.
        window (int) : The sample long window (For example: 1000)

    Returns:
        A list of indices which shows the boundaries.
    """
    try:
        word_env = envelope(word_sig, smooth)
        indices = rolling_window(function, word_env, window)
        return np.array(indices)
    except Exception as e:
        save_as_exception("Root", "get boundary", e)


def rolling_window(function, word_env, window):
    """
    A helper function which applying a defined function on a list.

     Args:
        function : A function which will be apply on enveloped word. For example: np.argmin.
        word_env : An array of enveloped signal for one word.
        window (int) : The sample long window (For example: 1000)

    Returns:
        A list of indices which meets the function.
    """
    le = window // 2
    ri = window - le
    i = le
    pos = []
    try:
        while i + ri <= len(word_env):
            if function(word_env[(i - le):(i + ri)]) == le:
                pos.append(i)
                i += ri
            else:
                i += 1
    except Exception as e:
        save_as_exception("Root", "rolling_window", e)
    return pos


def split_signal(word_signals, word_boundaries):
    """
     Split a digital signal to some parts based on the boundaries

     Args:
         word_signals : signal of words
         word_boundaries : indices of boundaries

     Returns:
         split signals.
     """
    word_tuples = zip(word_signals, word_boundaries)
    parts = []
    try:
        for wav, bound in word_tuples:
            if len(bound) > 0:
                parts.append(np.split(wav, bound))
            else:
                parts.append([wav])
        return parts
    except Exception as e:
        save_as_exception("Root", "Splitting", e)


def get_summary_freq_band(words_boundaries_parts, sample_rate):
    """
        A feature summary consists of frequency band number, the first intensity value, the median of
        all values in the frequency band, the minimum and maximum intensity, the last intensity value,
        and chunk index.

      Args:
          words_boundaries_parts : A list of all words which have been split based on a defined function.
          sample_rate (int) : Sample rate.

      Returns:
          A list of all FBSFs for all words in the Corpus.
      """

    fbsf = []
    for word in words_boundaries_parts:
        try:
            band_cues = []
            for index, part in enumerate(word, 1):
                spec = get_mel_spec(part, sample_rate)
                if spec is not None:
                    for ii, band in enumerate(spec, 1):
                        median = np.median(band)
                        new_median = int(median) if '0' in str(median) else median
                        band_cues.append('b{}start{}median{}min{}max{}end{}part{}'.
                                         format(ii, int(band[0]), new_median, int(band.min()), int(band.max()),
                                                int(band[-1]), index))

                else:
                    band_cues.append("tooShort")
                    save_as_exception("Root", "get summary frequency bands", "too short happened")

            fbsf.append('_'.join(band_cues))
        except Exception as e:
            save_as_exception("Root", "get summary frequency bands", e)
            pass

    return fbsf

