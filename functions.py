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


def rmse(actual, prediction):
    """
    calculates Root Mean Squared Error

    :param actual: vector containing actual log(prices + 1)
    :param prediction: vector containg predicted log(prices + 1)
    """
    return np.sqrt(np.mean((prediction - actual) ** 2))


rmse_scorer = make_scorer(rmse, greater_is_better=False)


def crossvalidate(data, model, scorer=rmsle_scorer, n=3):
    """
    calculates RMSLE and saves model info into a text file 'models/info.txt'

    :param data: full data set
    :param model: model object (eg. RandomForest())
    :param scorer: which scoring functions should be used
    :param n: how many times it crossvalidates
    :return e: mean of scores from every validation
    """

    scores = cross_val_score(model,
                             data.drop(['price', 'id'], axis=1).values,
                             data.price.values,
                             cv=n,
                             scoring=scorer)
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
    except Exception:
        return np.nan, np.nan, np.nan


def handle_missing(dataset):
    """
    changes nas to missing in columns category, brand_name and

    :param dataset:
    :return:
    """
    dataset.category1.fillna(value="missing", inplace=True)
    dataset.category2.fillna(value="missing", inplace=True)
    dataset.category3.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    return dataset
