
import time
import numpy as np
import xarray as xr
import pickle
from scipy import linalg
from itertools import chain
from pyndl import fast_corr
from rpy2.robjects import r, globalenv, pandas2ri


folds = pickle.load(open("./data/folds.p", "rb"))


def ldl_cv(data, semvecs, k=2):

    fold_stop = k
    events_all = data['wordtoken'].values
    n_events_all = len(events_all)

    for fold_number in range(1, fold_stop):
        fold_number = fold_number - 1

        fold = [event_ind - 1 for event_ind in folds[fold_number]]  # test
        fold_set = set(fold)  # test

        notfold = [event_ind for event_ind in range(n_events_all) if event_ind not in fold_set]  # train
        data_train = data.iloc[notfold, :]
        events_train = data_train['wordtoken'].values
        n_events_train = len(events_train)
        fbsfs_train = np.unique(list(chain.from_iterable([s.split("_") for s in data_train["FBSFs"]])))
        n_fbsfs_train = len(fbsfs_train)  # 39464
        fbsfs_map_train = {fbsf: ind for ind, fbsf in enumerate(fbsfs_train)}

        fc_train = np.zeros((n_events_train, n_fbsfs_train))
        ii = 0
        for index, row in data_train.iterrows():
            fbsfs_event = row.FBSFs.split("_")
            for fbsf in fbsfs_event:
                fc_train[ii, fbsfs_map_train[fbsf]] = 1.0
            ii += 1

        s_gold = np.zeros((n_events_train, n_dim))  # (118205, 4609)
        ii = 0
        for index, row in data_train.iterrows():
            s_gold[ii,] = semvecs.loc[row.wordtoken, :]
            ii += 1

        # test
        data_test = data.iloc[fold, :]

        events_test = data_test['wordtoken'].values
        n_events_test = len(events_test)  # 13167
        print('n_events_test:', n_events_test)

        events_test_unique = data_test['wordtoken'].unique()
        n_events_test_unique = len(events_test_unique)




        outcome_vectors = xr.DataArray(np.zeros((n_events_test_unique, n_dim)), dims=('outcomes', 'vector_dimensions'),
                                       coords={'outcomes': events_test_unique, 'vector_dimensions': vecdims})

        for ii in range(outcome_vectors.shape[0]):
            outcome_vectors.data[ii,] = semvecs.loc[events_test_unique[ii], :]

        fc_test = np.zeros((n_events_test, n_fbsfs_train))  # (13167, 39464)

        ii = 0
        for index, row in data_test.iterrows():  # 2 sec
            fbsfs_event = row.FBSFs.split("_")
            for fbsf in fbsfs_event:
                if fbsf in fbsfs_map_train:
                    fc_test[ii, fbsfs_map_train[fbsf]] = 1.0
            ii += 1





        squaredmatrix = fc_train.T @ fc_train  # 4 min # (39464, 39464)
        squaredmatrix_inv = linalg.pinvh(squaredmatrix)  # 3 hours and 23 minutes # (39464, 39464)
        fc_train_inv = squaredmatrix_inv @ fc_train.T  # 8 min # (39464, 118205) = (39464, 39464) * (39464, 118205)


        m1 = fc_train_inv @ s_gold
        s_hat = fc_test @ m1

        outcome_vectors2 = np.asfortranarray(outcome_vectors.data)
        ldl_shat = np.asfortranarray(s_hat)


        ldl_corrs = fast_corr.manual_corr(outcome_vectors2.T, ldl_shat.T)
        ldl_tp = sum([events_test[i] == events_test_unique[np.nanargmax(ldl_corrs[:, i])] for i in range(n_events_test)])
        print("ldl_tp: ", ldl_tp)
        print("LDL accuracy is: " + "{0:.2f}%".format((100 * ldl_tp) / n_events_test))



