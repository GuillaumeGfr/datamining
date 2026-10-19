import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#Configuration
seed = 42
mindf = 5
token = r'\b[^\W\d_]{2,}\b'
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

### Parameter sets ###
#Naive Bayes
k_values = [600, 700, 800, 900, 1000, 1100, 'all']

#Logistic regression
C_values = np.logspace(-3, 2, 20)

#Random forest
features = [2, 4, 6, 8, 10, 12]
trees = [100, 200, 300, 400, 500, 600, 700, 800]

#Gradient boosting
gb_param_grid = {
    'gb__learning_rate': [0.1, 0.2, 0.3, 0.4],
    'gb__max_depth': [5, 6, 7, 8],
}


def tune_naive_bayes(X, y):

    pipe = Pipeline([
        ('vec', CountVectorizer(min_df=mindf, token_pattern=token)),
        ('sel', SelectKBest(chi2)),
        ('nb', MultinomialNB()),
    ])

    search = GridSearchCV(pipe, {'sel__k': k_values}, cv=folds, n_jobs=-1)
    search.fit(X, y)

    print('Naive Bayes:', search.best_params_, search.best_score_)
    return search.best_estimator_

def tune_log_reg(X, y):

    pipe = Pipeline([
        ('vec', CountVectorizer(min_df=mindf, token_pattern=token)),
        ('logreg', LogisticRegression(l1_ratio = 1, solver = 'liblinear')),
    ])

    search = GridSearchCV(pipe, {'logreg__C': C_values}, cv=folds, n_jobs=-1)
    search.fit(X, y)

    print('Logistic Regression:', search.best_params_, search.best_score_)
    return search.best_estimator_

def tune_single_tree(X, y):

    vec = CountVectorizer(min_df=mindf, token_pattern=token)
    P = vec.fit_transform(X)

    path = DecisionTreeClassifier(random_state=seed).cost_complexity_pruning_path(P, y)
    alpha_values = path.ccp_alphas

    pipe = Pipeline([
        ('vec', CountVectorizer(min_df=mindf, token_pattern=token)),
        ('tree', DecisionTreeClassifier(criterion='gini', random_state=seed)),
    ])

    search = GridSearchCV(pipe, {'tree__ccp_alpha': alpha_values}, cv=folds, n_jobs = -1)
    search.fit(X, y)

    print('Decision Tree:', search.best_params_, search.best_score_)
    return search.best_estimator_

def tune_random_forests(X, y):

    results = []

    RF = RandomForestClassifier(oob_score = True, random_state=seed, n_jobs=-1)
    P = CountVectorizer(min_df=mindf, token_pattern=token).fit_transform(X)

    for feature in features:
        for tree in trees:
            RF.set_params(n_estimators=tree, max_features=feature)
            RF.fit(P, y)
            oob_error = 1 - RF.oob_score_
            results.append((feature, tree, oob_error))
            print(f'number of features={feature}, number of trees={tree}, oob error={oob_error}')

    best = min(results, key=lambda x: x[2])
    print('\nBest:', best)

    final_rf = Pipeline([
        ('vec', CountVectorizer(min_df=mindf, token_pattern=token)),
        ('rf', RandomForestClassifier(n_estimators=best[1], max_features=best[0], random_state=seed, n_jobs=-1)),
    ])
    final_rf.fit(X, y)
    return final_rf

def tune_gradient_boosting(X,y):

    pipe = Pipeline([
        ('vec', CountVectorizer(min_df=mindf, token_pattern=token)),
        ('gb', GradientBoostingClassifier(random_state=seed)),
    ])

    search = GridSearchCV(pipe, gb_param_grid, cv=folds, n_jobs=-1)
    search.fit(X, y)

    print('Gradient Boosting:', search.best_params_, search.best_score_)
    return search.best_estimator_





