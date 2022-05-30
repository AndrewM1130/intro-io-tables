# -*- coding: utf-8 -*-
"""
Created on Tue May  3 21:38:18 2022
"""

import json
import numpy as np
import pandas as pd

from econnet import data


mapping = json.load(open("data/category_mapping.json", 'r'))


def rolling_sub(x):
    return (x.iloc[-1] - x.iloc[0]) / x.iloc[0]


def max_diff(df, window):
    dif_dict = {}
    for ticker in df.columns.tolist():
        x = df[ticker]
        differences = x.rolling(window).apply(rolling_sub).dropna()
        max_dif = np.min(differences)
        idx = np.argmin(differences)
        dif_dict[ticker] = [idx, max_dif, differences]

    return dif_dict


def get_min_elements_index(array, n):
    arr_sort = np.sort(array)
    arr_sort = arr_sort[:n]
    index_min_elements = []
    for small_thing in arr_sort:
        index_small = np.where(array == small_thing)
        index_small = int(index_small[0])
        index_min_elements.append(index_small)

    return index_min_elements


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def equityDz(equity_df, window=18, cutoff_pct=-0.2):
    # equity_df is the equity index dataframe
    # window is an integer denoting the window for computing the difference
    # cutoff is the smallest drop we suspect is attributed to covid
    # returns dz

    # We're only considering COVID 19 at the moment, so year = 2020
    year = 2020
    H = data.I2IReqs(year).make_adjacency()

    dz = np.zeros(71)
    res = max_diff(equity_df, window)

    # this cutoff serves as the smallest difference we think is significant enough - like p values
    for key in res.keys():
        # if it's below a certain threshold - industry is not important to us
        if res[key][1] >= cutoff_pct:
            res[key][0] = 1e100

    # grab index of first relevant industry shift
    indices = [x[0] for x in res.values()]
    pandemic_start = min(indices)

    # calibrate dz for each industry at that window
    for key in res.keys():
        # grab difference for that industry at the pandemic
        # again set it to zero if no siginficant drop
        res[key].append(res[key][2][pandemic_start])

    # add the changes to the initial dz vector
    for key in res.keys():
        # find index match of dict key to H
        industry_idx = H.columns.tolist().index(key)

        # add the difference here to dz!
        dz[industry_idx] = res[key][3]

    # interpolate the industries that are not in a particular index
    # what industries are included in the intersection of the two?
    index_industries = equity_df.columns.tolist()
    missing_industries = list(set(H.index.tolist()) - set(index_industries))

    # for the industries not in, compute the covariance structure
    for missing_industry in missing_industries:
        cov_sum = 0
        for index_industry in index_industries:
            weight = np.cov(H.loc[missing_industry], H.loc[index_industry])[0, 1]  # want the off-diagonal cov
            dz[H.columns.tolist().index(missing_industry)] = weight * dz[H.columns.tolist().index(index_industry)]
            cov_sum += weight

        dz[H.columns.tolist().index(missing_industry)] = dz[H.columns.tolist().index(missing_industry)] / cov_sum

    # now we need to enforce the threshold
    for i in range(len(dz)):
        if dz[i] >= cutoff_pct:
            dz[i] = 0

    return pd.Series(dz, index=data.Table.industries)


def BEAdz(filePath="data/tzu-data/va-quarter.csv", quarter=1):
    """To distinguish between the initial shock and the higher-order effects,
    return by quarter"""

    ## Read in and create df_difference
    df_va_qtr = pd.read_csv(filePath, skiprows=5).iloc[:100, 1:]

    df_q1_q4 = df_va_qtr[['Q1', 'Q2', 'Q3', 'Q4']]
    df_q1_q4 = df_q1_q4.apply(pd.to_numeric, errors='coerce')

    idx = df_va_qtr.iloc[0:, 0].str.strip()
    df_q1_q4.index = idx
    df_q1_q4.rename(index=mapping, inplace=True)

    # Drop aggregate categories:
    dropenda = set(df_q1_q4.index) - set(data.Table.industries)
    df_q1_q4.drop(dropenda, axis=0, inplace=True)

    # multiple categories map to same thing, just combine them:
    df_q1_q4 = df_q1_q4.groupby(level=0).sum()

    df_difference = df_q1_q4.diff(axis=1)
    shocks = df_difference.apply(np.min, axis=1)
    shocks_index = df_difference.apply(np.argmin, axis=1)

    # NOTE: we want relative to the pandemic onset, ie Q1
    # min_quarter = []
    # for i in range(df_q1_q4.shape[0]):
    #     min_quarter.append(df_q1_q4.iloc[i, shocks_index[i]])

    # shocks_rel = np.divide(shocks, min_quarter)

    # NOTE: Everything relative to the onset of the pandemic: Q1
    shocks_rel = 1. - np.divide(df_q1_q4.iloc[:, quarter], df_q1_q4.Q1)

    ## if pos diff, then impute with 0
    shocks_rel[shocks_rel > 0] = 0

    # match = set(shocks_rel.index).intersection(set(data.Table.industries))
    # shocks_rel2 = shocks_rel[match]

    # only government stuff missing
    for extra in set(data.Table.industries) - set(shocks_rel.index):
        shocks_rel.loc[extra] = 0

    # correct order
    shocks_rel = shocks_rel.loc[data.Table.industries]

    return shocks_rel


if __name__ == "__main__":
    dz = BEAdz(quarter=0)
