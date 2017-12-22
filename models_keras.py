import os

import numpy as np
import pandas as pd
from keras import Input, Model, optimizers
from keras.layers import Dense, Embedding, GRU, concatenate, Flatten, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasRegressor

from Kaggle_Mercari.functions import crossvalidate, rmse_scorer

# read data
os.chdir("C:\\kaggle_mercari")
train = pd.read_csv("keras_train.tsv", sep="\t")
print("data read")

MAX_NAME_SEQ = 20  # 17
MAX_ITEM_DESC_SEQ = 60  # 269
MAX_CATEGORY_NAME_SEQ = 20  # 8
MAX_TEXT = np.max([train.seq_name.max(),
                   train.seq_category_name.max(),
                   train.seq_item_description.max()]
                  ) + 2
MAX_CATEGORY = train.category.max() + 1
MAX_BRAND = train.brand.max() + 1
MAX_CONDITION = train.item_condition_id.max() + 1


# KERAS DATA DEFINITION
def get_keras_data(dataset):
    x = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(dataset.seq_item_description,
                                   maxlen=MAX_ITEM_DESC_SEQ),
        'brand': np.array(dataset.brand),
        'category': np.array(dataset.category),
        'category_name': pad_sequences(dataset.seq_category_name,
                                       maxlen=MAX_CATEGORY_NAME_SEQ),
        'item_condition': np.array(dataset.item_condition_id),
        'num_vars': np.array(dataset[["shipping"]])
    }
    return x


X_train = get_keras_data(train)


def get_model():
    # params
    dr_r = 0.25

    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand = Input(shape=[1], name="brand")
    category = Input(shape=[1], name="category")
    category_name = Input(shape=[X_train["category_name"].shape[1]],
                          name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

    # Embeddings layers
    emb_size = 60

    emb_name = Embedding(MAX_TEXT, emb_size // 3)(name)
    emb_item_desc = Embedding(MAX_TEXT, emb_size)(item_desc)
    emb_category_name = Embedding(MAX_TEXT, emb_size // 3)(category_name)
    emb_brand = Embedding(MAX_BRAND, 10)(brand)
    emb_category = Embedding(MAX_CATEGORY, 10)(category)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)

    rnn_layer1 = GRU(16)(emb_item_desc)
    rnn_layer2 = GRU(8)(emb_category_name)
    rnn_layer3 = GRU(8)(emb_name)

    # main layer
    main_l = concatenate([
        Flatten()(emb_brand),
        Flatten()(emb_category),
        Flatten()(emb_item_condition),
        rnn_layer1,
        rnn_layer2,
        rnn_layer3,
        num_vars
    ])
    main_l = Dropout(dr_r)(Dense(128)(main_l))
    main_l = Dropout(dr_r)(Dense(32)(main_l))

    output = Dense(1, activation="linear")(main_l)

    model = Model([name,
                   item_desc,
                   brand,
                   category,
                   category_name,
                   item_condition,
                   num_vars],
                  output)

    optimizer = optimizers.Adam()
    model.compile(loss="mse",
                  optimizer=optimizer)
    return model


model = get_model()

model.fit(X_train,
          train.price,
          epochs=2,
          batch_size=1024,
          verbose=False)

# def baseline_model():
#     model = Sequential()
#     model.add(Embedding(input_dim=MAX_TEXT, output_dim=64, ))
#     model.compile(loss='mean_squared_error',
#                   optimizer='adam')
#     return model
#
#
# scikit_keras = KerasRegressor(build_fn=baseline_model, epochs=16, batch_size=256, verbose=False)
#
# print(crossvalidate(train, scikit_keras, rmse_scorer, 3))
