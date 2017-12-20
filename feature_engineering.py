"""
in this file we prepare data for building models
"""

import os

import pandas as pd

from Kaggle_Mercari.functions import split_column

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

# Merging train and test in order to process them together

# changing name of columns train_id and test_id to the same name - id
train = train.rename(columns={'train_id': 'id'})
test = test.rename(columns={'test_id': 'id'})

# marking which data comes from train set
train['is_train'] = 1
test['is_train'] = 0

# actual merger
train_test_combined = pd.concat([train.drop(['price'], axis=1), test], axis=0)

# make every text column lowercase because text mining
train_test_combined.loc[:, ["name",
                            "category_name",
                            "brand_name",
                            "item_description"]] = train_test_combined.loc[:,
                                                   ["name",
                                                    "category_name",
                                                    "brand_name",
                                                    "item_description"]].apply(lambda x: x.astype(str).str.lower())

# splitting categories into 3
train_test_combined['category1'], train_test_combined['category2'], train_test_combined['category3'] = zip(
    *train_test_combined["category_name"].apply(split_column))

# changing data types from object to category
train_test_combined.name = train_test_combined.name.astype('category')
train_test_combined.brand_name = train_test_combined.brand_name.astype('category')
train_test_combined.category1 = train_test_combined.category1.astype('category')
train_test_combined.category2 = train_test_combined.category2.astype('category')
train_test_combined.category3 = train_test_combined.category3.astype('category')

train_test_combined.name = train_test_combined.name.cat.codes
train_test_combined.brand_name = train_test_combined.brand_name.cat.codes
train_test_combined.category1 = train_test_combined.category1.cat.codes
train_test_combined.category2 = train_test_combined.category2.cat.codes
train_test_combined.category3 = train_test_combined.category3.cat.codes

# dropping unused columns - item_description just for now
train_test_combined = train_test_combined.drop(['item_description'], axis=1)
train_test_combined = train_test_combined.drop(['category_name'], axis=1)

# splitting train and test data again
df_train = train_test_combined.loc[train_test_combined['is_train'] == 1]
df_test = train_test_combined.loc[train_test_combined['is_train'] == 0]

# is_train is no longer needed
df_train = df_train.drop(['is_train'], axis=1)
df_test = df_test.drop(['is_train'], axis=1)

# adding price again the train
df_train['price'] = train.price
print(df_train.head())
df_train.to_csv("df_train.tsv", sep="\t")
