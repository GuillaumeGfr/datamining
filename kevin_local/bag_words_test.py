# %%
# SECTION 
# testing bag of words cleaning and characterization. Consider: bigrams, lowercase, lemitization (room/rooms), sparse terms

# start from pre-filtered english reviews
# Clean everything and split to training and test set in later steps. 

import pandas as pd

df = pd.read_csv("../data/data_eng.csv", index_col=0)
print(df.head())

# %%
# Confirm the provided NA flags match the real missing values
print(df[["na_up_review", "na_down_review"]].dtypes)
print((df["na_up_review"] == df["Upside_Review"].isna()).all())
print((df["na_down_review"] == df["Downside_Review"].isna()).all())


# %%
# Noted there are some errors in downside reviews
up_mismatch = df[df["na_up_review"] != df["Upside_Review"].isna()]
down_mismatch = df[df["na_down_review"] != df["Downside_Review"].isna()]

print(len(up_mismatch), len(down_mismatch))
print(up_mismatch[["Upside_Review", "na_up_review", "source"]].head(20))
print(down_mismatch[["Downside_Review", "na_down_review", "source"]].head(20))

# %%
up_empty = df["Upside_Review"].fillna("").str.strip() == ""
down_empty = df["Downside_Review"].fillna("").str.strip() == ""

print((df["na_up_review"] == up_empty).all())
print((df["na_down_review"] == down_empty).all())

# %%
still_off = df[df["na_down_review"] != down_empty]
print(len(still_off))
print(still_off[["Downside_Review", "na_down_review", "source"]].apply(
    lambda col: col.map(repr) if col.name == "Downside_Review" else col))

# %%
# compute own flags
df["up_empty"] = df["Upside_Review"].fillna("").str.strip() == ""
df["down_empty"] = df["Downside_Review"].fillna("").str.strip() == ""

# %%
# NOTE: using recomputed flags: provided na_* flags disagreed with the text in 5 AI rows

# Fill, combine, clean
# Normalize : replace nulls with empty strings so pandas can join, clean, and vectorize
df[["Upside_Review", "Downside_Review"]] = df[["Upside_Review", "Downside_Review"]].fillna("")

# Build clean text column: lowercase, clean up whitespace
df["text"] = (df["Upside_Review"] + " " + df["Downside_Review"]) \
    .str.lower() \
    .str.replace(r"\s+", " ", regex=True) \
    .str.strip()

print(df.head(3))

# %%

# Characterization checks. NOTE: using self computed fields after noting mismatch
# 0 = human, 1 = AI
print(df["source"].value_counts())                                  
print(df.groupby("source")[["up_empty", "down_empty"]].mean())
print(pd.crosstab(df["source"], df["Sentiment"]))

# %%
# count words per review and compare the distributions
df["n_words"] = df["text"].str.split().str.len()
print(df.groupby(["source", "Sentiment"])["n_words"].describe())

# %%
# findings:
#   AI reviews seems to have a more standardized length. Humans write short reviews. 
#   Only exception is a few humans write more when they give a bad review, AIs don't care.
#   While mean for negative human reviews is longer its only few long human reviews dragging the mean up.
#   Median shows still in general they are shorter.

#   "Length is not a feature in your bag-of-words model directly, but it influences the counts: longer reviews have larger counts across the board, so the models can partly pick up on length.
#   For question 3, these results suggest a hypothesis worth stating before you test it: since the classes differ more in positive reviews, the random forest may separate fake and genuine positive reviews more easily. The McNemar-style test you run later will tell you whether that holds.
#   The histogram should show this clearly. The human distributions will have a tall peak at short lengths and a long tail to the right; the AI distributions will be narrower bumps further right. Because of the long human tail, the set_xlim cutoff may be useful, or a log scale for the x-axis."

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
bins = range(0, df["n_words"].max() + 10, 10)

for ax, sentiment in zip(axes, ["POS", "NEG"]):
    subset = df[df["Sentiment"] == sentiment]
    ax.hist(subset.loc[subset["source"] == 0, "n_words"], bins=bins, alpha=0.6, label="Human")
    ax.hist(subset.loc[subset["source"] == 1, "n_words"], bins=bins, alpha=0.6, label="AI")
    ax.set_title(f"{sentiment} reviews")
    ax.set_xlabel("Words per review")
    ax.legend()

axes[0].set_ylabel("Number of reviews")
plt.tight_layout()
plt.savefig("../figures/review_length_hist.pdf")
plt.show()

# %%
## SECTION BAG OF WORDS AND VOCABULARY

# FIXME split between training and test set, WILL NEED TO USE DATASET FROM BOBBY OR GG or just use this
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df, test_size=0.25, stratify=df["source"], random_state=42
)

# %%
from sklearn.feature_extraction.text import CountVectorizer

# fit_transform on the training text does two things: it learns the vocabulary (every distinct word that passes your filters)
#   , then counts each word in each review. 
#   transform on the test text only counts; it uses the training vocabulary, and words never seen in training are ignored. 
#   The results are sparse matrices with one row per review and one column per vocabulary word.
#   NOTE: naming convention is capital letters matrices and lowercase vector
#   min_df is important, it controls sparse term removal. A value of 5 means that only words appearing in at least 5 reviews are kept.

#   POSSIBLE EDITS 
#   keep stop words (e.g., the, was, very) as they might be indicators of AI writing?
#   to experiment with bigrams add something like ngram_range=(1, 2)
#   referencing hotel and city names could be AI indicator but adds complication 

# NOTE: tokenization default ignores punctuation and drops single character tokens. 
vectorizer = CountVectorizer(min_df=5)
X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])
y_train, y_test = train_df["source"], test_df["source"]

# %%
vocab = vectorizer.get_feature_names_out()

# this is the number of words in our vocabulary
print(len(vocab))
print(vocab[:50])

# Most frequent words per class
import numpy as np
counts = pd.DataFrame({
    # the np.asarray is housekeeping since we stored as a sparse matrix
    "human": np.asarray(X_train[y_train.values == 0].sum(axis=0)).ravel(),
    "ai":    np.asarray(X_train[y_train.values == 1].sum(axis=0)).ravel(),
}, index=vocab)
print(counts.sort_values("ai", ascending=False).head(5))
print(counts.sort_values("human", ascending=False).head(5))

# %%
# NOTE: there are a few numerical responses. It is a choice whether to keep them.
#   It appears they are stronlgy included in human reviews, so they may indicate a non-AI review (i.e., an AI may not complain about a hidden 5 euro charge)

nums = [w for w in vocab if w.isdigit()]
print(counts.loc[nums])

# %%
# FIXME Cleaning decisions to finalize
# Numbers: keep, replace with a placeholder, or remove, based on the per-class check.
# Hotel and city names: keep and discuss, or remove.
# Stopwords: keep (my recommendation) or remove, with a reason.
# Sparse-term threshold: a fixed min_df, or tuned as a hyperparameter later.
# N-grams: single words only, or single words plus bigrams.
# Stemming or lemmatization: these merge word forms like "staff"/"staffs" or "recommend"/"recommended." 
#   It's optional; skipping it is fine if we say so.

# put the final cleaning steps into a shared function or script
#   , for example preprocessing.py with a clean_text() function and a fixed split with a recorded random_state. 
#   That way all five models use identical data and the same train/test split. 
#   The assignment needs this, both for fair comparisons and for the McNemar tests, which pair predictions on the same test reviews. 
#   It also fits the README requirement to explain which script produces what.

