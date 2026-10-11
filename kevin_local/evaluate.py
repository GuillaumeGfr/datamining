"""Test-set metrics and McNemar's test."""
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate(y_true, y_pred):
    """Metrics with AI-generated (label 1) as the positive class."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1),
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "f1": f1_score(y_true, y_pred, pos_label=1),
    }


def mcnemar(y_true, pred_a, pred_b, name_a="A", name_b="B"):
    """McNemar's test with continuity correction (Dietterich, 1998).
    Returns the 2x2 table of correct/incorrect and the p-value."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    table = pd.DataFrame(
        [[np.sum(a_ok & b_ok), np.sum(a_ok & ~b_ok)],
         [np.sum(~a_ok & b_ok), np.sum(~a_ok & ~b_ok)]],
        index=[f"{name_a} correct", f"{name_a} wrong"],
        columns=[f"{name_b} correct", f"{name_b} wrong"])
    n01, n10 = table.iloc[0, 1], table.iloc[1, 0]
    if n01 + n10 == 0:
        return table, 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = chi2.sf(stat, df=1)
    return table, p_value
