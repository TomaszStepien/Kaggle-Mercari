import pandas as pd
import numpy as np
import os
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score
# from scipy.stats import spearmanr, pearsonr


def error(actual, prediction):
    '''calculates Root Mean Squared Logarithmic Error given 2 vectors'''
    return np.sqrt(np.mean((np.log(prediction + 1) - np.log(actual + 1)) ** 2))


def crossvalidate(data, model, features, test_size=0.3):
    '''calculates crossvalidation error'''
    iterations = int(np.floor(1 / test_size))
    nrows = data.shape[0]
    rows = np.array(range(nrows))
    errors = []
    for i in range(iterations - 1):
        r = np.random.choice(rows, int(test_size * nrows), replace=False)
        rows = np.setdiff1d(rows, r)
        train = data[~data.index.isin(r)]
        test = data.iloc[r, ]

        model.fit(train.loc[:, features], train.loc[:, "price"])
        predicted = model.predict(test.loc[:, features])
        e = error(test.loc[:, "price"], predicted)
        errors.append(e)

    train = data[~data.index.isin(rows)]
    test = data.iloc[rows, ]

    model.fit(train.loc[:, features], train.loc[:, "price"])
    predicted = model.predict(test.loc[:, features])
    e = error(test.loc[:, "price"], predicted)
    errors.append(e)

    e = np.mean(errors)
    # save model data to text file
    if "models" not in os.listdir(os.getcwd()):
        os.mkdir("models")

    name = "models\\" + str(np.round(e, 6)) + '_' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + ".txt"
    info = open(name, mode='w')
    info.write("mean RMSLE: " + str(e) + '\n')
    info.write("features:\n")
    for f in features:
        info.write(f)
        info.write('\n')

    info.close()

    return e


# read data
os.chdir("C:\\kaggle_mercari")
sweaters = pd.read_csv("sweaters.tsv", sep="\t")

colnames = list(sweaters.columns.values)
FEATURES = [f for f in colnames if "MODEL" in f]

forest = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0)

t = crossvalidate(sweaters, forest, FEATURES)

print(t)
