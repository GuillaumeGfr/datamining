# %%
# all model code here
# one function per classifier (e.g. train_naive_bayes)
# at each save run jupytext --sync models_kr.ipynb

# FIXME CLAUDE DUMP NEEDS FIXING

# %%
"""One function per classifier.

Every train_* function has the same signature:
    train_xxx(X_train, y_train)  ->  (fitted_model, info)

  X_train : list/Series of raw review texts
  y_train : array of 0/1 labels
  fitted_model : a sklearn Pipeline (vectorizer + classifier), already refit on
                 the full training set with the selected hyperparameters, so it can
                 be called directly as fitted_model.predict(X_test_texts)
  info : dict with the selected hyperparameters and the validation score

Because the vectorizer is inside the pipeline, it is re-fit within every CV fold,
so the vocabulary never sees the validation fold (and never sees the test set).
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import sklearn

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def make_vectorizer():
    """Shared bag-of-words settings, so all models use the same preprocessing.
    min_df=0.01 drops terms that occur in fewer than 1% of the training reviews
    (the equivalent of removing sparse terms)."""
    return CountVectorizer(lowercase=True, min_df=0.01)


def make_lasso_logreg():
    """L1-penalised logistic regression. scikit-learn 1.8 replaced penalty='l1' by
    l1_ratio=1; this works on both old and new versions."""
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=5000)
    return LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000)


def _grid_search(pipeline, param_grid, X_train, y_train):
    search = GridSearchCV(pipeline, param_grid, cv=CV, scoring="accuracy",
                          n_jobs=-1, refit=True)  # refit=True -> retrain on all training data
    search.fit(X_train, y_train)
    info = {"best_params": search.best_params_,
            "cv_accuracy": search.best_score_,
            "cv_results": search.cv_results_}
    return search.best_estimator_, info


# --------------------------------------------------------------------------- #
# 1. Multinomial naive Bayes  (hyperparameter: k in top-k feature selection)
# --------------------------------------------------------------------------- #
def train_naive_bayes(X_train, y_train):
    pipe = Pipeline([
        ("vect", make_vectorizer()),
        ("select", SelectKBest(chi2)),
        ("clf", MultinomialNB()),
    ])
    grid = {"select__k": [25, 50, 100, 200, 400, "all"]}
    return _grid_search(pipe, grid, X_train, y_train)


# --------------------------------------------------------------------------- #
# 2. Logistic regression with Lasso penalty  (hyperparameter: C = 1/lambda)
# --------------------------------------------------------------------------- #
def train_logistic_lasso(X_train, y_train):
    pipe = Pipeline([
        ("vect", make_vectorizer()),
        ("clf", make_lasso_logreg()),
    ])
    grid = {"clf__C": np.logspace(-2, 2, 13)}
    return _grid_search(pipe, grid, X_train, y_train)


# --------------------------------------------------------------------------- #
# 3. Single classification tree  (hyperparameter: cost-complexity alpha)
# --------------------------------------------------------------------------- #
def train_tree(X_train, y_train):
    pipe = Pipeline([
        ("vect", make_vectorizer()),
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])
    grid = {"clf__ccp_alpha": [0.0, 0.001, 0.002, 0.004, 0.008, 0.015, 0.03]}
    return _grid_search(pipe, grid, X_train, y_train)


# --------------------------------------------------------------------------- #
# 4. Random forest  (hyperparameters: n_estimators, max_features)
#    Tuned with out-of-bag accuracy instead of cross-validation.
# --------------------------------------------------------------------------- #
def train_random_forest(X_train, y_train):
    vect = make_vectorizer()
    X_bow = vect.fit_transform(X_train)   # unsupervised step, no labels used
    n_features = X_bow.shape[1]

    best_score, best_params, results = -np.inf, None, []
    for n_trees in [200, 500]:
        for max_feat in ["sqrt", "log2", 0.1, 0.3]:
            rf = RandomForestClassifier(n_estimators=n_trees, max_features=max_feat,
                                        oob_score=True, n_jobs=-1,
                                        random_state=RANDOM_STATE)
            rf.fit(X_bow, y_train)
            results.append({"n_estimators": n_trees, "max_features": max_feat,
                            "oob_accuracy": rf.oob_score_})
            if rf.oob_score_ > best_score:
                best_score = rf.oob_score_
                best_params = {"n_estimators": n_trees, "max_features": max_feat}

    final = Pipeline([
        ("vect", make_vectorizer()),
        ("clf", RandomForestClassifier(**best_params, n_jobs=-1,
                                       random_state=RANDOM_STATE)),
    ])
    final.fit(X_train, y_train)
    info = {"best_params": best_params, "oob_accuracy": best_score,
            "n_features": n_features, "oob_results": results}
    return final, info


# --------------------------------------------------------------------------- #
# 5. Gradient boosting  (hyperparameters: learning rate, tree depth, n trees)
# --------------------------------------------------------------------------- #
def train_gradient_boosting(X_train, y_train):
    pipe = Pipeline([
        ("vect", make_vectorizer()),
        ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])
    grid = {
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__max_depth": [1, 2, 3],
        "clf__n_estimators": [100, 300],
    }
    return _grid_search(pipe, grid, X_train, y_train)


# Registry: the notebook loops over this. Add/remove/rename models here only.
MODELS = {
    "Naive Bayes": train_naive_bayes,
    "Logistic (Lasso)": train_logistic_lasso,
    "Classification tree": train_tree,
    "Random forest": train_random_forest,
    "Gradient boosting": train_gradient_boosting,
}

