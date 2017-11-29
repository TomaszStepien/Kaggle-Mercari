import pandas as pd
import numpy as np
import os
from datetime import datetime


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor

# from sklearn.metrics import r2_score
# from scipy.stats import spearmanr, pearsonr


def error(actual, prediction):
    """calculates Root Mean Squared Logarithmic Error given 2 vectors"""
    return np.sqrt(np.mean((np.log(prediction + 1) - np.log(actual + 1)) ** 2))


def crossvalidate(data, model, features, test_size=0.3):
    """calculates crossvalidation error"""
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
    info.write("\nfeatures:\n")
    for f in features:
        info.write(f)
        info.write('\n')
    info.write("\nparameters:\n")
    params = model.get_params()
    for key in params:
        info.write(key + ": " + str(params[key]) + "\n")
    info.close()

    return e


# read data
os.chdir("C:\\kaggle_mercari")
sweaters = pd.read_csv("sweaters.tsv", sep="\t")

# select features
colnames = list(sweaters.columns.values)
FEATURES = [f for f in colnames if "MODEL" in f]

# define model
MODEL1 = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0)
MODEL2 = GradientBoostingRegressor()
MODEL3 = SGDRegressor()

# magia
for MODEL in (MODEL1, MODEL2, MODEL3):
    t = crossvalidate(sweaters, MODEL, FEATURES)
    print(t)
