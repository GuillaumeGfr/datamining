# %%
# Kevin version of code for running models on test data
# at each save run jupytext --sync run_models_kr.ipynb

# FIXME Claude dump

# %%
# Fake review detection: train and evaluate all models
# The model functions live in `models.py` (one `train_*` function per classifier). 
# This notebook loads the data, makes the fixed train/test split, trains (or loads) each model, 
#   and evaluates them on the held-out test set.
# Flow is data_prep -> models -> evaluate with run_models calling each one. 

# %load_ext autoreload
# autoreload picks up edits to the .py files without restarting the kernel. for quick testing
# %autoreload 2

from pathlib import Path
import joblib
import pandas as pd

# FIXME all these names may not be accurate
from data_prep import load_reviews, make_split
from models import MODELS
from evaluate import evaluate, mcnemar

## 1. Data and stratified 75/25 split
# these are the first pass pre-processing steps
df = load_reviews("../data/all_data.csv")
train_df, test_df = make_split(df)

X_train, y_train = train_df["text"], train_df["label"].values
X_test,  y_test  = test_df["text"],  test_df["label"].values

print(f"train: {len(train_df)}  test: {len(test_df)}")
print("AI share train/test:", y_train.mean().round(3), y_test.mean().round(3))

## 2. Train every model (hyperparameters tuned on the training set only)
# Set `RETRAIN = False` to load the saved models instead of tuning again. 
# Run it once with RETRAIN = True, then switch to False while working on evaluation 
#   or feature importance. Switch back to True whenever anything changes in models.py or data_prep.py. 
#   The saved files don’t know code changed, so with False we keep getting the old models.

RETRAIN = True
SAVE_DIR = Path("saved_models")
SAVE_DIR.mkdir(exist_ok=True)

# fitted will hold trained models, tuning_info holds selected hyperparameters and validation scores
fitted, tuning_info = {}, {}

for name, train_fn in MODELS.items():
    path = SAVE_DIR / f"{train_fn.__name__}.joblib"
    if RETRAIN or not path.exists():
        print(f"Training {name}...")
        model, info = train_fn(X_train, y_train)
        joblib.dump((model, info), path)
    else:
        model, info = joblib.load(path)
    fitted[name], tuning_info[name] = model, info
    print(f"  {name}: {info['best_params']}")

## 3. Evaluate on the test set: run every trained model on test set and store predictions
predictions = {name: model.predict(X_test) for name, model in fitted.items()}
# this is equivalant to code below. By storing we avoid calling predict multiple times and ensures
#   that McNemar step uses same prediction.

# predictions = {}
# for name, model in fitted.items():
#   predictions[name] = model.predict(X_test)

# table of results
results = pd.DataFrame({name: evaluate(y_test, pred) for name, pred in predictions.items()}).T
results.round(3)
# this might work to make it table for report results.round(3).to_latex()

## 4. McNemar tests (alpha = 0.01)
# run McNemar test on each of the pairs below to compare accuracy for statistical significance.
comparisons = [
    ("Naive Bayes", "Logistic (Lasso)"),        # relevant question 1, generative vs. descriminative linear
    ("Classification tree", "Random forest"),     # relevant to question 2, do ensembles beat single tree?
    ("Classification tree", "Gradient boosting"), # relevant question 2
]
for a, b in comparisons:
    table, p = mcnemar(y_test, predictions[a], predictions[b], a, b)
    print(f"\n{a} vs {b}:  p = {p:.4f}")
    display(table)
    # inside a loop only explicit display shows table
    # assignment significance level is alpha = .01
    # include these tables with pvalues in report

## 5. Use this to run a single model on its own (can be used while working on one function 
#   and don't want to retrain all models)
#  

# from models import train_logistic_lasso

# model, info = train_logistic_lasso(X_train, y_train)
# print(info["best_params"], round(info["cv_accuracy"], 3))
# evaluate(y_test, model.predict(X_test))
