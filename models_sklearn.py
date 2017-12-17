import pandas as pd
import numpy as np
import os
from datetime import datetime

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer


def RMSLE(actual, prediction):
    """calculates Root Mean Squared Logarithmic Error given 2 vectors"""
    return np.sqrt(np.mean((np.log(prediction + 1) - np.log(actual + 1)) ** 2))


RMSLE_errors = make_scorer(RMSLE, greater_is_better=False)


def crossvalidate(data, model, iterations=3):
    """calculates crossvalidation error"""

    scores = cross_val_score(model,
                             data.drop(['price'], axis=1),
                             data.price, cv=iterations,
                             scoring=RMSLE_errors)

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


# read data
os.chdir("C:\\kaggle_mercari")
sweaters = pd.read_csv("df_train.tsv", sep="\t")
print("data read")

# set up model
MODEL = GradientBoostingRegressor()
print("model set up")

# magia
# t = crossvalidate(sweaters, MODEL)
# print("Error: ", t)

# PARAMETER TUNING

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    sweaters.drop(['price'], axis=1), sweaters.price, test_size=0.4, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'loss': ['ls'],
                     'n_estimators': [100],
                     'max_depth': [1, 2, 3]},
                    {'loss': ['lad'],
                     'n_estimators': [100],
                     'max_depth': [1, 2, 3]}]

print("# Tuning hyper-parameters for RMSLE")

clf = GridSearchCV(GradientBoostingRegressor(), tuned_parameters, cv=5,
                   scoring=RMSLE_errors)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
