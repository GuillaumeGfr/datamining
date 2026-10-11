"""Shared data preparation: load the English reviews, build the cleaned text,
and make the fixed stratified train/test split. Imported by run_models.ipynb
and by the exploration notebook, so every model uses identical data."""
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42  # keep identical across the team so everyone gets the same split


def clean_text(upside, downside):
    """Combine upside and downside parts into one lowercase string with tidy whitespace."""
    return ((upside.fillna("") + " " + downside.fillna(""))
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip())


def load_reviews(path="../data/data_eng.csv"):
    """Load the pre-filtered English reviews and add the columns the models need:
    text, label (0 = human, 1 = AI) and sentiment (POS/NEG).
    The original columns are kept for exploration."""
    df = pd.read_csv(path, index_col=0)

    # Recomputed empty flags: the provided na_* flags disagree with the text in 5 AI rows
    df["up_empty"] = df["Upside_Review"].fillna("").str.strip() == ""
    df["down_empty"] = df["Downside_Review"].fillna("").str.strip() == ""

    df["text"] = clean_text(df["Upside_Review"], df["Downside_Review"])
    df["label"] = df["source"].astype(int)
    df["sentiment"] = df["Sentiment"]
    df["n_words"] = df["text"].str.split().str.len()
    return df


def make_split(df, test_size=0.25, random_state=RANDOM_STATE):
    """Stratified split on the label. Returns train_df, test_df."""
    return train_test_split(df, test_size=test_size, stratify=df["label"],
                            random_state=random_state)
