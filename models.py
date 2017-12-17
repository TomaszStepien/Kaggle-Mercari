import pandas as pd
import numpy as np
import os
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer


def RMSLE(actual, prediction):
    """calculates Root Mean Squared Logarithmic Error given 2 vectors"""
    return np.sqrt(np.mean((np.log(prediction + 1) - np.log(actual + 1)) ** 2))


RMSLE_erros = make_scorer(RMSLE, greater_is_better=False)


def crossvalidate(data, model, iterations=3):
    """calculates crossvalidation error"""

    scores = cross_val_score(model,
                             data.drop(['price'], axis=1),
                             data.price, cv=iterations,
                             scoring=RMSLE_erros)

    e = abs(np.mean(scores))
    # save model data to text file
    if "models" not in os.listdir(os.getcwd()):
        os.mkdir("models")

    name = "models\\" + str(np.round(e, 6)) + '_' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + ".txt"
    info = open(name, mode='w')
    info.write("mean RMSLE: " + str(e) + '\n')
    # info.write("\nfeatures:\n")
    # for f in features:
    #     info.write(f)
    #     info.write('\n')
    info.write("\nparameters:\n")
    params = model.get_params()
    for key in params:
        info.write(key + ": " + str(params[key]) + "\n")
    info.close()

    return e


# read data
os.chdir("C:\\kaggle_mercari")
sweaters = pd.read_csv("df_train.tsv", sep="\t")

# select features
# colnames = list(sweaters.columns.values)
# FEATURES = [f for f in colnames if "MODEL" in f]

# define model
MODEL1 = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0)
MODEL2 = GradientBoostingRegressor()
MODEL3 = SGDRegressor()

# magia
for MODEL in (MODEL1, MODEL2, MODEL3):
    t = crossvalidate(sweaters, MODEL)
    print(t)
