import os

import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor

from Kaggle_Mercari.functions import crossvalidate

# read data
os.chdir("C:\\kaggle_mercari")
sweaters = pd.read_csv("df_train.tsv", sep="\t")
print("data read")


def baseline_model():
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=sweaters.shape[1] - 2))
    model.add(Dense(units=1, activation='softmax'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam')
    return model


b_m = baseline_model()
# b_m.fit(sweaters.drop(['price', 'id'], axis=1).values, sweaters.price.values, batch_size=32, epochs=5, verbose=False)
print("model compiled")

scikit_keras = KerasRegressor(build_fn=baseline_model, epochs=1, batch_size=32, verbose=False)

print(crossvalidate(sweaters, scikit_keras, 3))
