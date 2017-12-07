import os
import pandas as pd
#import numpy as np
from unidecode import unidecode
from collections import Counter

# read data - descriptions of the items
os.chdir("C:\\kaggle_mercari")
descriptions = pd.read_csv("descriptions.tsv", sep="\t")

#getting rid of items with no descriptions
descriptions['item_description'] = descriptions["item_description"][descriptions["item_description"]!="no description yet"]
descriptions = descriptions.dropna()

#how many rows do we take info account
#n = 200000

#choosing a subset of the dataframe
#part_desc = descriptions[:n]

#if we are not subsetting the data
part_desc = descriptions

# combining all the descpritions into one huge string
wordstring = ''.join(str(unidecode(s)) for s in part_desc['item_description'])

#print(wordstring)

#getting rid of everything that is not a letter or a space
wordstring = ''.join([i for i in wordstring if (i.isalpha() or i.isspace())])
#print('wordstring')

#NO+ is an important information, let's save it:
#now instead of "no damage" we will have "nondamage" just to see the whole thing
wordstring = wordstring.replace(" no "," non")
wordstring = wordstring.replace(" not "," not")

# creating a list out of the string
wordlist = wordstring.split()

#getting rid of unimportant words:
junk_words = ["and","or","if","is","are","a","in","it","i","to","of","the","on","be","for","with","have"]

wordlist2 = []
[wordlist2.append(w) for w in wordlist if w not in junk_words] 
# now the wordlist2 consists of only words that not on the junk list

#renaming the list without stupid words as the original one
wordlist = wordlist2


#counting the words
word_freq = Counter(wordlist)

# list of words without duplicated
unique_words = [w for w in word_freq.most_common()]

# creating a dataframe so we can put it into csv later
dm = pd.DataFrame(data = unique_words)
dm.columns = ["word","freq"]

print(dm.head(3))

dm = dm[:200]

print(result.head(25))

#importing the data into csv
#dm.to_csv("word_frequency.csv", columns = ["word","freq"], index = False)