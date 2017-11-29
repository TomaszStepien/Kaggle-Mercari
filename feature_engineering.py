import os
import pandas as pd

# read data
os.chdir("C:\\kaggle_mercari")
sweaters = pd.read_csv("train.tsv", sep="\t")

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
sweaters = sweaters.rename(index=str, columns={"shipping" : "MODEL_shipping"})

# make dummy variables from condition id
dm = pd.get_dummies(sweaters.loc[:, "item_condition_id"], dummy_na=True, prefix="MODEL_condition")
sweaters = pd.concat([sweaters, dm], axis=1)

# save new data
sweaters.to_csv("sweaters.tsv", sep="\t", encoding="utf-8")
