import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor


def error(actual, prediction):
    '''calculates Root Mean Squared Logarithmic Error given 2 vectors'''
    return np.sqrt(np.mean((np.log(prediction + 1) - np.log(actual + 1)) ** 2))


def crossvalidate(data, model, features, test_size=0.3):
    '''calculates crossvalidation error'''
    it = 1 / test_size
    n = data.shape[0]
    rows = np.array(range(n))
    errors = []
    for i in range(int(it) - 1):
        r = np.random.choice(rows, int(test_size * n), replace=False)
        rows = np.setdiff1d(rows, r)
        train = data[~data.index.isin(r)]
        test = data.iloc[r, ]

        model.fit(train.loc[:, features], train.loc[:, "price"])
        p = model.predict(test.loc[:, features])
        e = error(test.loc[:, "price"], p)
        errors.append(e)

    train = data[~data.index.isin(rows)]
    test = data.iloc[rows, ]

    model.fit(train.loc[:, features], train.loc[:, "price"])
    p = model.predict(test.loc[:, features])
    e = error(test.loc[:, "price"], p)
    errors.append(e)

    return e


# read data
os.chdir("C:\\kaggle_mercari")
sweaters = pd.read_csv("sweaters.tsv", sep="\t")

l = list(sweaters.columns.values)
FEATURES = [f for f in l if "MODEL" in f]

forest = RandomForestRegressor(max_depth=2, random_state=0)

t = crossvalidate(sweaters, forest, FEATURES)

print(t)
