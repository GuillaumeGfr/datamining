# preprocessing/variables.py
# Shared data for all model scripts. No prints or plots, so importing stays silent.
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data_eng.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.25
REMOVE_NAMES = True   # drop each review's own hotel and city name from the text

BOW_PARAMS = {
    "lowercase": True,
    "stop_words": None,                 # keeps "the", "was", "very"
    "min_df": 5,                        # sparse-term removal
    "ngram_range": (1, 1),              # (1, 2) adds bigrams
    "token_pattern": r"(?u)\b\w\w+\b",  # keeps numbers (human signal)
}

# 1. load English reviews (data_eng.csv = all_data.csv filtered on Review_Language)
df = pd.read_csv(DATA_PATH, index_col=0)

# 2. own empty flags: provided na_down_review is wrong for 5 AI rows
df[["Upside_Review", "Downside_Review"]] = df[["Upside_Review", "Downside_Review"]].fillna("")
df["up_empty"] = df["Upside_Review"].str.strip() == ""
df["down_empty"] = df["Downside_Review"].str.strip() == ""

# 3. text: upside + downside, lowercase, newlines and repeated spaces collapsed
df["text"] = (df["Upside_Review"] + " " + df["Downside_Review"]).str.lower()
if REMOVE_NAMES:
    df["text"] = [t.replace(h.lower(), " ") for t, h in zip(df["text"], df["Hotel Name"])]
    df["text"] = [t.replace(c.lower(), " ") for t, c in zip(df["text"], df["City Name"])]
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
df["n_words"] = df["text"].str.split().str.len()

# 4. stratified 75/25 split, whole rows kept so Sentiment is available for question 3
strata = df["source"].astype(str) + "_" + df["Sentiment"]
train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, stratify=strata, random_state=RANDOM_STATE
)
y_train = train_df["source"]   # 0 = human, 1 = AI
y_test = test_df["source"]

# 5. bag of words: vocabulary from training text only
vectorizer = CountVectorizer(**BOW_PARAMS)
X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])
vocab = vectorizer.get_feature_names_out()

# 6. word totals per class in the training set (Kevin's characterization)
counts = pd.DataFrame({
    "human": np.asarray(X_train[y_train.values == 0].sum(axis=0)).ravel(),
    "ai": np.asarray(X_train[y_train.values == 1].sum(axis=0)).ravel(),
}, index=vocab)
nums = [w for w in vocab if w.isdigit()]

# 7 Quick validation split of the training set (1200 / 300), for tinkering only

X_tr, X_val, y_tr, y_val = train_test_split(
X_train, y_train, test_size=0.2, stratify=y_train, random_state=0
)
