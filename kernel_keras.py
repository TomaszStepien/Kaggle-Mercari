import numpy as np
import pandas as pd
from keras import backend as K
from keras import optimizers
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

# define data types
types = {'train_id': 'int64',
         'item_condition_id': 'int8',
         'price': 'float64',
         'shipping': 'int8'}

# read data
train = pd.read_csv('../input/train.tsv', sep='\t', low_memory=True, dtype=types)
test = pd.read_csv('../input/test.tsv', sep='\t', low_memory=True, dtype=types)

train['target'] = np.log1p(train['price'])
# In[ ]:


# HANDLE MISSING VALUES
print("Handling missing values...")


def handle_missing(dataset):
    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    return dataset


train = handle_missing(train)
test = handle_missing(test)
print(train.shape)
print(test.shape)

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
train.head(3)

# EMBEDDINGS MAX VALUE
MAX_NAME_SEQ = 20  # 17
MAX_ITEM_DESC_SEQ = 60  # 269
MAX_CATEGORY_NAME_SEQ = 20  # 8
MAX_TEXT = np.max([np.max(train.seq_name.max()),
                   np.max(test.seq_name.max()),
                   np.max(train.seq_category_name.max()),
                   np.max(test.seq_category_name.max()),
                   np.max(train.seq_item_description.max()),
                   np.max(test.seq_item_description.max())]) + 2
MAX_CATEGORY = np.max([train.category.max(), test.category.max()]) + 1
MAX_BRAND = np.max([train.brand.max(), test.brand.max()]) + 1
MAX_CONDITION = np.max([train.item_condition_id.max(),
                        test.item_condition_id.max()]) + 1


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
X_test = get_keras_data(test)

# KERAS MODEL DEFINITION


dr = 0.25


def get_model():
    # params
    dr_r = dr

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
    # main_l = Dropout(dr_r)(Dense(128)(main_l))
    main_l = Dropout(dr_r)(Dense(32)(main_l))

    # output
    output = Dense(1, activation="linear")(main_l)

    # model
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


def exp_decay(init, fin, steps):
    return (init / fin) ** (1 / (steps - 1)) - 1


# FITTING THE MODEL
epochs = 1
BATCH_SIZE = 512 * 3
steps = int(len(X_train['name']) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.013, 0.009
lr_decay = exp_decay(lr_init, lr_fin, steps)

model = get_model()
K.set_value(model.optimizer.lr, lr_init)
K.set_value(model.optimizer.decay, lr_decay)

history = model.fit(X_train, train.target,
                    epochs=epochs,
                    batch_size=BATCH_SIZE,
                    verbose=False
                    )

# CREATE PREDICTIONS
preds = model.predict(X_test)
preds = np.expm1(preds)
X_test['price'] = preds

X_test[['test_id', 'price']].to_csv('output.csv', index=False)
