import os

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from Kaggle_Mercari.functions import rmsle_scorer

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

# PARAMETER TUNING - gridsearch

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
                   scoring=rmsle_scorer)
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
