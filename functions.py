"""
in this file we declare functions used in main files
"""

import os
from datetime import datetime

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score


def rmsle(actual, prediction):
    """
    calculates Root Mean Squared Logarithmic Error

    :param actual: vector containing actual prices
    :param prediction: vector containg predicted prices
    """
    return np.sqrt(np.mean((np.log(prediction + 1) - np.log(actual + 1)) ** 2))


rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


def crossvalidate(data, model, n=3):
    """
    calculates RMSLE and saves model info into a text file 'models/info.txt'

    :param data: full data set
    :param model: model object (eg. RandomForest())
    :param n: how many times it crossvalidates
    :return e: mean of scores from every crossvalidation
    """

    scores = cross_val_score(model,
                             data.drop(['price'], axis=1),
                             data.price, cv=n,
                             scoring=rmsle_scorer)
    e = abs(np.mean(scores))

    # save model data to a text file
    if "models" not in os.listdir(os.getcwd()):
        os.mkdir("models")

    name = "models\\" + str(np.round(e, 6)) + '_' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + ".txt"
    info = open(name, mode='w')
    info.write("mean RMSLE: " + str(e) + '\n')
    info.write("\nfeatures:\n")
    for f in list(data.columns.values):
        info.write(f)
        info.write('\n')
    info.write("\nparameters:\n")
    params = model.get_params()
    for key in params:
        info.write(key + ": " + str(params[key]) + "\n")
    info.close()

    return e


def split_column(column):
    """
    splits a column based on '/'

    :param column: column to be separated
    :return: a tuple of columns
    """

    try:
        new_column1, new_column2, new_column3 = column.split('/')
        return new_column1, new_column2, new_column3
    except:
        return np.nan, np.nan, np.nan
