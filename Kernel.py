import os
import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor

# read data
# kaggle format
#sweaters = pd.read_csv('../input/train.tsv', sep='\t')
# local format
os.chdir("C:\\kaggle_mercari")
sweaters = pd.read_csv("train.tsv", sep="\t") 

print(sweaters.head())
print(sweaters.info())

# make every text column lowercase because text mining
sweaters.loc[:, ["name",
                 "category_name",
                 "brand_name",
                 "item_description"]] = sweaters.loc[:,
                                        ["name",
                                         "category_name",
                                         "brand_name",
                                         "item_description"]].apply(lambda x: x.astype(str).str.lower())

# rename shipping column cause it can go to the model already
sweaters = sweaters.rename(index=str, columns={"shipping": "MODEL_shipping"})

# make dummy variables from condition id
dm = pd.get_dummies(sweaters.loc[:, "item_condition_id"], dummy_na=True, prefix="MODEL_condition").astype(np.int8)
sweaters = pd.concat([sweaters, dm], axis=1)

# make dummies from brand names (counts < threshold go to other)
sweaters["brand_name_trimmed"] = sweaters["brand_name"]
counts = sweaters.loc[:, "brand_name"].value_counts()
brands = counts[counts > 15000].index
select = ~sweaters["brand_name_trimmed"].isin(list(brands))
sweaters.loc[select, "brand_name_trimmed"] = "other"

dm = pd.get_dummies(sweaters.loc[:, "brand_name_trimmed"], dummy_na=False, prefix="MODEL_brand").astype(np.int8)
sweaters = pd.concat([sweaters, dm], axis=1)

# make dummies from category_name
sweaters["MODEL_men"] = 0
select = sweaters["category_name"].str.contains("men")
sweaters.loc[select, "MODEL_men"] = 1

sweaters["MODEL_women"] = 0
select = sweaters["category_name"].str.contains("women")
sweaters.loc[select, "MODEL_women"] = 1
sweaters.loc[select, "MODEL_men"] = 0

sweaters["MODEL_beauty"] = 0
select = sweaters["category_name"].str.contains("beauty")
sweaters.loc[select, "MODEL_beauty"] = 1

sweaters["MODEL_makeup"] = 0
select = sweaters["category_name"].str.contains("makeup")
sweaters.loc[select, "MODEL_makeup"] = 1

# save new data
#print(sweaters.head())

#train, test = train_test_split(sweaters, test_size=0.3)

colnames = list(sweaters.columns.values)
FEATURES = [f for f in colnames if "MODEL" in f]

model = SGDRegressor()
sweaters.info()

model.fit(sweaters.loc[:, FEATURES], sweaters.loc[:, "price"])

#Read testing data
# kaggle format
# test = pd.read_csv('../input/test.tsv', sep='\t') - 
# local format
os.chdir("C:\\kaggle_mercari")
test = pd.read_csv('test.tsv', sep='\t')

# make every text column lowercase because text mining
test.loc[:, ["name",
                 "category_name",
                 "brand_name",
                 "item_description"]] = test.loc[:,
                                        ["name",
                                         "category_name",
                                         "brand_name",
                                         "item_description"]].apply(lambda x: x.astype(str).str.lower())

# rename shipping column cause it can go to the model already
test = test.rename(index=str, columns={"shipping": "MODEL_shipping"})

# make dummy variables from condition id
dm = pd.get_dummies(test.loc[:, "item_condition_id"], dummy_na=True, prefix="MODEL_condition").astype(np.int8)
test = pd.concat([test, dm], axis=1)

# make dummies from brand names (counts < threshold go to other)
test["brand_name_trimmed"] = test["brand_name"]
select = ~test["brand_name_trimmed"].isin(list(brands))
test.loc[select, "brand_name_trimmed"] = "other"

dm = pd.get_dummies(test.loc[:, "brand_name_trimmed"], dummy_na=False, prefix="MODEL_brand").astype(np.int8)
test = pd.concat([test, dm], axis=1)

# make dummies from category_name
test["MODEL_men"] = 0
select = test["category_name"].str.contains("men")
test.loc[select, "MODEL_men"] = 1

test["MODEL_women"] = 0
select = test["category_name"].str.contains("women")
test.loc[select, "MODEL_women"] = 1
test.loc[select, "MODEL_men"] = 0

test["MODEL_beauty"] = 0
select = test["category_name"].str.contains("beauty")
test.loc[select, "MODEL_beauty"] = 1

test["MODEL_makeup"] = 0
select = test["category_name"].str.contains("makeup")
test.loc[select, "MODEL_makeup"] = 1

test['price'] = model.predict(test.loc[:,FEATURES])
#print(test.head())

test[['test_id','price']].to_csv('output.csv', index=False)
