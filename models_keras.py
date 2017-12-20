from keras.models import Sequential
from keras.layers import Dense
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# read data
os.chdir("C:\\kaggle_mercari")
sweaters = pd.read_csv("df_train.tsv", sep="\t")
print("data read")

X_train, X_test, y_train, y_test = train_test_split(
    sweaters.drop(['price'], axis=1), sweaters.price, test_size=0.4, random_state=0)

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=7))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='softmax'))


model.compile(loss='mean_squared_error',
              optimizer='adam')

model.fit(X_train.values, y_train.values, epochs=5, batch_size=32)
