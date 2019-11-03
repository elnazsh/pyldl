
"""
This file contains all functions to do LDL auditory comprehension.

@author: Elnaz Shafaei-Bajestan, Masoumeh Moradipour Tari
@date: April 2019
@place: Germany, University of Tuebingen
"""

import xarray as xr
import numpy as np
import scipy
from itertools import chain
from pyndl import fast_corr


def predict(data, word_form, cues, semvecs, n_fbsfs, fbsfs_map, n_events, n_vec_dims):
    """
        This function predict the word meaning base on the acoustic cues.

      Args:
          data : train and test together
          word_form: the name of the column in data that specifies word forms
          cues: the name of the column in data that specifies acoustic cues
          semvecs : the golden semantic vectors
          n_fbsfs: number of unique acoustic cues
          fbsfs_map : a mapping between acosutic cues and the token
          n_events : number of tokens.
          n_vec_dims : number of dimension in semantic vectors

      Returns:
         A predicted word after learning.
      """
    s = get_s_matrix(data, word_form, semvecs, n_events, n_vec_dims)
    fc = get_fc_matrix(data, cues, n_events, n_fbsfs, fbsfs_map)
    fc_inv = get_fc_inv(fc)
    w = get_w_matrix(fc_inv, s)
    return fc @ w


def get_correlation(outcome_vectors, s_hat):
    """
        This function computes the correlation between predicted words and the golden words.

      Args:
          outcome_vectors : the golden semantic vectors
          s_hat : the predicted semantic vectors
      Returns:
         A correlation matrix between predicted and the golden semantic vectors.
      """

    outcome_vectors2 = np.asfortranarray(outcome_vectors.data)
    ldl_shat = np.asfortranarray(s_hat)
    return fast_corr.manual_corr(outcome_vectors2.T, ldl_shat.T)


def get_accuracy(corrs, events, n_events, outcomes):
    """
        This function computes the accuracy using the correlation.

      Args:
        corrs : the correlation matrix.
        events : all tokens in the data.rds
        n_events : number of tokens.
        outcomes : all types in the data.rds
      Returns:
         accuracy of the model.
    """

    ldl_tp = sum([events[i] == outcomes[np.nanargmax(corrs[:, i])] for i in range(n_events)])
    return (100 * ldl_tp) / n_events


def save_correlation_matrix(ldl_corrs, events, output):
    """ This is a helper function"""

    corrs_temp = xr.DataArray(data=ldl_corrs, dims=('events1', 'events2'), coords={'events1': events, 'events2': events})
    corrs_temp.to_netcdf(output)
    del corrs_temp


def get_outcome_vectors(semvecs, outcomes, n_outcomes, vecdims, n_vec_dims):
    """ This is a helper function"""
    outcome_vectors = xr.DataArray(np.zeros((n_outcomes, n_vec_dims)), dims=('outcomes', 'vector_dimensions'),
                                   coords={'outcomes': outcomes, 'vector_dimensions': vecdims})

    for ii in range(outcome_vectors.shape[0]):
        outcome_vectors.data[ii,] = semvecs.loc[outcomes[ii], :]
    return outcome_vectors


def get_fc_inv(fc):
    """ This is a helper function"""
    return scipy.linalg.pinvh(fc.T @ fc) @ fc.T


def save_fc_inv(fc_inv, events, fbsfs, output):
    """ This is a helper function"""
    fc_inv_save = xr.DataArray(data=fc_inv, dims=('fbsfs', 'events'), coords={'fbsfs': fbsfs, 'events': events})
    fc_inv_save.to_netcdf(output)
    del fc_inv_save


def get_w_matrix(fc_inv, s):
    """ This is a helper function"""
    return fc_inv @ s

def get_fc_matrix(data, cues, n_events, n_fbsfs, fbsfs_map):
    """ This is a helper function"""

    fc = np.zeros((n_events, n_fbsfs))
    ii = 0
    for index, row in data.iterrows():
        fbsfs_event = row[cues].split("_")
        for fbsf in fbsfs_event:
            fc[ii, fbsfs_map[fbsf]] = 1.0        
        ii += 1
    return fc


def get_s_matrix(data, word_form, semvecs, n_events, n_vec_dims):
    """ This is a helper function"""

    s = np.zeros((n_events, n_vec_dims))
    ii = 0
    for index, row in data.iterrows():
        s[ii, ] = semvecs.loc[row[word_form], :]
        ii += 1
    return s


def get_cues_outcomes(data, word_form, cues, sep="_"):
    """
        Args:
            data: a pandas dataframe
            word_form: the name of the column in data that specifies word forms
            cues: the name of the column in data that specifies acoustic cues
        Returns:
            a tuple of 3 elements:
                a list of events
                a list of outcomes
                a list of cues
    """
    events = data[word_form].values
    outcomes = data[word_form].values
    cues = np.unique(list(chain.from_iterable(data[cues].apply(lambda x: x.split(sep)))))
    return events, outcomes, cues
