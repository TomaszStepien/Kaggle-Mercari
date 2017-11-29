import pandas as pd
# import numpy as np
import os

os.chdir("C:\\kaggle_mercari")

sweaters = pd.read_csv("sweaters.tsv", sep="\t")

# print(sweaters.head())

# brand_name
# print all unique values with counts
counts = sweaters.loc[:, "brand_name"].value_counts()
# print(counts.iloc[:300,])

print(counts[counts > 500])

# there are 4809 unique brand names (4810 including nans)
# probably should be trimmed -
# brands with counts lower than some threshold should go to "other" category
