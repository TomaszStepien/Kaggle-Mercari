import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

# define data types
types_train = {'train_id': 'int64',
               'item_condition_id': 'int8',
               'price': 'float64',
               'shipping': 'int8'}

types_test = {'test_id': 'int64',
              'item_condition_id': 'int8',
              'price': 'float64',
              'shipping': 'int8'}

# read data
train = pd.read_csv('../input/train.tsv', sep='\t', low_memory=True, dtype=types_train)
test = pd.read_csv('../input/test.tsv', sep='\t', low_memory=True, dtype=types_test)

# Merging train and test in order to process them together

# changing name of columns train_id and test_id to the same name - id
train = train.rename(columns={'train_id': 'id'})
test = test.rename(columns={'test_id': 'id'})

# marking which data comes trom train set
train['is_train'] = 1
test['is_train'] = 0

# actual merger
train_test_combined = pd.concat([train.drop(['price'], axis=1), test], axis=0)

# make every text column lowercase because text mining
train_test_combined.loc[:, ["name",
                            "category_name",
                            "brand_name",
                            "item_description"]] = test.loc[:,
                                                   ["name",
                                                    "category_name",
                                                    "brand_name",
                                                    "item_description"]].apply(lambda x: x.astype(str).str.lower())


#
def category_name(category):
    """splits categories from sth/sth/sth to 3 columns"""
    try:
        category1, category2, category3 = category.split('/')
        return category1, category2, category3
    except:
        return np.nan, np.nan, np.nan


train_test_combined['category1'], train_test_combined['category2'], train_test_combined['category3'] = zip(
    *train_test_combined["category_name"].apply(category_name))

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

# print(train_test_combined.head())
# print(train_test_combined.info())

# splitting train and test data again
df_train = train_test_combined.loc[train_test_combined['is_train'] == 1]
df_test = train_test_combined.loc[train_test_combined['is_train'] == 0]

# is_train is no longer needed
df_train = df_train.drop(['is_train'], axis=1)
df_test = df_test.drop(['is_train'], axis=1)

# adding price again the train
df_train['price'] = train.price

# Fitting the model
X = df_train.drop(['price', 'id'], axis=1)
y = df_train.price

model = GradientBoostingRegressor(n_estimators=500, max_depth=4, loss='lad')

model.fit(X, y)

# predicting on the test set
preds = model.predict(df_test.drop(['id'], axis=1))
df_test['price'] = preds
df_test['price'] = df_test['price'].apply(lambda x: x if x > 0 else 0)
df_test = df_test.rename(columns={'id': 'test_id'})

# saving the output
df_test[['test_id', 'price']].to_csv('output.csv', index=False)
