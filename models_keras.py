import os

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor

from Kaggle_Mercari.functions import crossvalidate, rmsle_scorer, rmse_scorer

# read data
os.chdir("C:\\kaggle_mercari")
sweaters = pd.read_csv("df_train.tsv", sep="\t")
print("data read")


def baseline_model():
    model = Sequential()
    model.add(Dense(units=7, activation='tanh', input_dim=sweaters.shape[1] - 2))
    model.add(Dense(units=14, activation='relu'))
    model.add(Dense(units=28, activation='relu'))
    model.add(Dense(units=56, activation='relu'))
    model.add(Dense(units=112, activation='relu'))
    model.add(Dense(units=56, activation='relu'))
    model.add(Dense(units=28, activation='relu'))
    model.add(Dense(units=14, activation='relu'))
    model.add(Dense(units=7, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam')
    return model


scikit_keras = KerasRegressor(build_fn=baseline_model, epochs=16, batch_size=256, verbose=False)

print(crossvalidate(sweaters, scikit_keras, rmse_scorer, 3))
