# %%
# EVALUATE FILE
# Computation of four performance measures (accuracy, precision, recall, F1 score)
# Run McNemar, proscribed signifigance test

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Take true test labels, one model's predictions, and return a dictionary with 4 sklearn metrics.
# To be called once for each of the the five final models.
def evaluate(y_true, y_pred):
    # Metrics with AI-generated (label 1) as the positive class.
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1),
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "f1": f1_score(y_true, y_pred, pos_label=1),
    }


# comparing models, seek disgreements in results, for questions 1 and 2 with alpha of .01
# both models scored on same 500 test reviews so results are paired
# use slide method of McNemar as an exact binomal test (slide 43 of log reg slides)
# Under H0 (equal accuracy), each disagreement is a fair coin flip, so the smaller 
#   off-diagonal count ~ Binomial(n01 + n10, 0.5). Returns the 2x2 table 
#   of correct/incorrect and the two-sided p-value.

def mcnemar(y_true, pred_a, pred_b, name_a="A", name_b="B"):
    # standardize as numpy arrays
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    table = pd.DataFrame(
        [[np.sum(a_ok & b_ok), np.sum(a_ok & ~b_ok)],
         [np.sum(~a_ok & b_ok), np.sum(~a_ok & ~b_ok)]],
        # for dev testing
        index=[f"{name_a} correct", f"{name_a} wrong"],
        columns=[f"{name_b} correct", f"{name_b} wrong"])
    # strip labels, exact=true means binomial dist false is chi square approximation
    result = sm_mcnemar(table.values, exact=True)
    # include table in Overleaf, p-value to be compared to the alpha. 
    return table, result.pvalue

# unpack via table, p = mcnemar(y_test, pred_nb, pred_lr, name_a="NB", name_b="LR")

