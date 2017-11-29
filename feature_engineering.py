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
sweaters = sweaters.rename(index=str, columns={"shipping": "MODEL_shipping"})

# make dummy variables from condition id
dm = pd.get_dummies(sweaters.loc[:, "item_condition_id"], dummy_na=True, prefix="MODEL_condition")
sweaters = pd.concat([sweaters, dm], axis=1)

# make dummies from brand names (counts < threshold go to other)
sweaters["brand_name_trimmed"] = sweaters["brand_name"]
counts = sweaters.loc[:, "brand_name"].value_counts()
brands = counts[counts < 15000].index
select = sweaters["brand_name_trimmed"].isin(list(brands))
sweaters.loc[select, "brand_name_trimmed"] = "other"

dm = pd.get_dummies(sweaters.loc[:, "brand_name_trimmed"], dummy_na=False, prefix="MODEL_brand")
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
print(sweaters.head())
sweaters.to_csv("sweaters.tsv", sep="\t", encoding="utf-8")
