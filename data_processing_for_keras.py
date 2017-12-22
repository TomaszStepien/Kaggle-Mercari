import os

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

os.chdir("C:\\Kaggle_Mercari")

# define data types
types_train = {'train_id': 'int64',
               'item_condition_id': 'int8',
               'price': 'float64',
               'shipping': 'int8'}

types_test = {'test_id': 'int64',
              'item_condition_id': 'int8',
              'price': 'float64',
              'shipping': 'int8'}

train = pd.read_csv('train.tsv', sep='\t', low_memory=True, dtype=types_train)
test = pd.read_csv('test.tsv', sep='\t', low_memory=True, dtype=types_test)

# HANDLE MISSING VALUES
print("Handling missing values...")


def handle_missing(dataset):
    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    return dataset


train = handle_missing(train)
test = handle_missing(test)

# PROCESS CATEGORICAL DATA
print("Handling categorical variables...")
le = LabelEncoder()

le.fit(np.hstack([train.category_name, test.category_name]))
train['category'] = le.transform(train.category_name)
test['category'] = le.transform(test.category_name)

le.fit(np.hstack([train.brand_name, test.brand_name]))
train['brand'] = le.transform(train.brand_name)
test['brand'] = le.transform(test.brand_name)
del le, train['brand_name'], test['brand_name']

# PROCESS TEXT: RAW
print("Text to seq process...")
print("   Fitting tokenizer...")

raw_text = np.hstack([train.category_name.str.lower(),
                      train.item_description.str.lower(),
                      train.name.str.lower()])

tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)
print("   Transforming text to seq...")
train["seq_category_name"] = tok_raw.texts_to_sequences(train.category_name.str.lower())
test["seq_category_name"] = tok_raw.texts_to_sequences(test.category_name.str.lower())
train["seq_item_description"] = tok_raw.texts_to_sequences(train.item_description.str.lower())
test["seq_item_description"] = tok_raw.texts_to_sequences(test.item_description.str.lower())
train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())

train.price = np.log1p(train.price)
print(train.head(6))

print(train.dtypes)

train.to_csv("keras_train.tsv", sep='\t')

# MAX_NAME_SEQ = 20  # 17
# MAX_ITEM_DESC_SEQ = 60  # 269
# MAX_CATEGORY_NAME_SEQ = 20  # 8
# MAX_TEXT = np.max([np.max(train.seq_name.max())
#                       , np.max(test.seq_name.max())
#                       , np.max(train.seq_category_name.max())
#                       , np.max(test.seq_category_name.max())
#                       , np.max(train.seq_item_description.max())
#                       , np.max(test.seq_item_description.max())]) + 2
# MAX_CATEGORY = np.max([train.category.max(), test.category.max()]) + 1
# MAX_BRAND = np.max([train.brand.max(), test.brand.max()]) + 1
# MAX_CONDITION = np.max([train.item_condition_id.max(),
#                         test.item_condition_id.max()]) + 1
#
#
# # KERAS DATA DEFINITION
# def get_keras_data(dataset):
#     X = {
#         'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ)
#         , 'item_desc': pad_sequences(dataset.seq_item_description
#                                      , maxlen=MAX_ITEM_DESC_SEQ)
#         , 'brand': np.array(dataset.brand)
#         , 'category': np.array(dataset.category)
#         , 'category_name': pad_sequences(dataset.seq_category_name
#                                          , maxlen=MAX_CATEGORY_NAME_SEQ)
#         , 'item_condition': np.array(dataset.item_condition_id)
#         , 'num_vars': np.array(dataset[["shipping"]])
#     }
#     return X
#
#
# X_train = get_keras_data(train)
# X_test = get_keras_data(test)
