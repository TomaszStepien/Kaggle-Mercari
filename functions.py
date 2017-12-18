"""
in this file we declare functions later used in main files
"""

import os
from datetime import datetime

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score


def rmsle(actual, prediction):
    """calculates Root Mean Squared Logarithmic Error given 2 vectors"""
    return np.sqrt(np.mean((np.log(prediction + 1) - np.log(actual + 1)) ** 2))


rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


def crossvalidate(data, model, iterations=3):
    """calculates RMSLE and saves model info into a text file"""

    scores = cross_val_score(model,
                             data.drop(['price'], axis=1),
                             data.price, cv=iterations,
                             scoring=rmsle_scorer)

    e = abs(np.mean(scores))
    # save model data to text file
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
