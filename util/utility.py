"""
This file contains all helper functions.

@author: Masoumeh Moradipour Tari
@date: September 2019
@place: Germany, University of Tuebingen
"""

import pyreadr
import pandas as pd
import xarray as xr
from Exception.exception import save_as_exception


def read_rds_file(path):
    """
    Reading a rds file using the pyreadr package. It contains all words in the the corpus.
    Args:
        path: The path of the file, Example: './data/words.rds'
    Returns:
        pandas.DataFrame with columns wordtoken, start, end, File, Prev, FBSFs, triphones
    """

    try:
        result = pyreadr.read_r(path)
        return result[None]
    except Exception as e:
        print(type(e))
        save_as_exception(path, "words.rds", e)


def read_pkl_file(path):
    """
    Reading a pkl file using the Pandas package.
    Args:
        path: The path of the file, Example: './data/words.pkl'
    Returns:
        pandas.DataFrame
    """

    try:
        return pd.read_pickle(path)
    except Exception as e:
        print(type(e))
        save_as_exception("root", path, e)


def read_nc_file(path):

    """
    Reading a nc file using the xarray package.
    Args:
        path: The path of the file, Example: './data/words.nc'
    Returns:
        xnarray
    """

    try:
        return xr.open_dataarray(path)
    except Exception as e:
        print(type(e))
        save_as_exception("root", path, e)