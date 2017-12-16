import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# define data types
types_train = {'train_id': 'int64',
             'item_condition_id': 'int8',
             'price': 'float64',
             'shipping': 'int8'}

# read data
train = pd.read_csv('../input/train.tsv', sep='\t', low_memory=True, dtype=types_train)

# local format 
# os.chdir("C:\\kaggle_mercari")
# sweaters = pd.read_csv("train.tsv", sep="\t", low_memory=True, dtype=types_train) 

# make every text column lowercase because text mining
train.loc[:, ["name",
                 "category_name",
                 "brand_name",
                 "item_description"]] = train.loc[:,
                                        ["name",
                                         "category_name",
                                         "brand_name",
                                         "item_description"]].apply(lambda x: x.astype(str).str.lower())

types_test = {'test_id': 'int64',
             'item_condition_id': 'int8',
             'price': 'float64',
             'shipping': 'int8'}
             
test = pd.read_csv('../input/test.tsv', sep='\t', low_memory=True, dtype=types_test)

# make every text column lowercase because text mining
test.loc[:, ["name",
             "category_name",
             "brand_name",
             "item_description"]] = test.loc[:,
                                        ["name",
                                         "category_name",
                                         "brand_name",
                                         "item_description"]].apply(lambda x: x.astype(str).str.lower())
                                         

#print(train.head())
#print(test.head())

# Merging train and test in order to process them together

# changing name of columns train_id and test_id to the same name - id
train = train.rename(columns = {'train_id' : 'id'})
test = test.rename(columns = {'test_id' : 'id'})

# marking which data comes trom train set
train['is_train'] = 1
test['is_train'] = 0

# actual merger
train_test_combined = pd.concat([train.drop(['price'],axis =1),test],axis = 0)

# spliting categories from sth/sth/sth to 3 columns

def category_name(category):
    try:
        Category1, Category2, Category3 = category.split('/')
        return Category1, Category2, Category3
    except:
        return np.nan, np.nan, np.nan
        
train_test_combined['Category1'], train_test_combined['Category2'], train_test_combined['Category3'] = zip(*train_test_combined["category_name"].apply(category_name))

# print(train_test_combined.head())
# print(train_test_combined.head())

# print(train_test_combined.info())
# changing data types from object to category
train_test_combined.name = train_test_combined.name.astype('category')
train_test_combined.brand_name = train_test_combined.brand_name.astype('category')
train_test_combined.Category1 = train_test_combined.Category1.astype('category')
train_test_combined.Category2 = train_test_combined.Category2.astype('category')
train_test_combined.Category3 = train_test_combined.Category3.astype('category')

train_test_combined.name = train_test_combined.name.cat.codes
train_test_combined.brand_name = train_test_combined.brand_name.cat.codes
train_test_combined.Category1 = train_test_combined.Category1.cat.codes 
train_test_combined.Category2 = train_test_combined.Category2.cat.codes
train_test_combined.Category3 = train_test_combined.Category3.cat.codes

# dropping usused columns - item_description just for now
train_test_combined = train_test_combined.drop(['item_description'],axis = 1)
train_test_combined = train_test_combined.drop(['category_name'],axis = 1)

#print(train_test_combined.head())
#print(train_test_combined.info())

# spliting train and test data again
df_train = train_test_combined.loc[train_test_combined['is_train']==1]
df_test = train_test_combined.loc[train_test_combined['is_train']==0]

# is_train is no longer needed
df_train = df_train.drop(['is_train'],axis=1)
df_test = df_test.drop(['is_train'],axis=1)

# adding price again the train 
df_train['price'] = train.price

# Fitting the model
X =  df_train.drop(['price'],axis =1)
y =  df_train.price

model = GradientBoostingRegressor(n_estimators=150, max_depth=4)
    
model.fit(X, y)

# predicting on the test set
preds = model.predict(df_test)
df_test['price'] = preds

# saving the output
df_test[['id','price']].to_csv('output.csv', index=False)

