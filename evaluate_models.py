import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Configuration (same as in tuning_functions.py)
seed = 42
mindf = 5
token = r'\b[^\W\d_]{2,}\b'
ngram = (1,2)

# label of the class counted as "positive" for precision/recall/F1
# check the possible labels with: print(y_train.unique())
positive_label = 'ai'


### Load data ###
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df['text'] = train_df['text'].fillna('')
test_df['text'] = test_df['text'].fillna('')

X_train, y_train = train_df['text'], train_df['source']
X_test, y_test = test_df['text'], test_df['source']


### Load tuned parameters ###
with open('best_params.json') as f:
    params = json.load(f)


### Build the models with the tuned parameters ###
def make_vectorizer():
    return CountVectorizer(min_df=mindf, token_pattern=token, ngram_range=ngram)

nb = Pipeline([
    ('vec', make_vectorizer()),
    ('sel', SelectKBest(chi2)),
    ('nb', MultinomialNB()),
])
nb.set_params(**params['naive_bayes'])

lr = Pipeline([
    ('vec', make_vectorizer()),
    ('logreg', LogisticRegression(l1_ratio=1, solver='liblinear')),
])
lr.set_params(**params['logistic_regression'])

tree = Pipeline([
    ('vec', make_vectorizer()),
    ('tree', DecisionTreeClassifier(criterion='gini', random_state=seed)),
])
tree.set_params(**params['decision_tree'])

rf = Pipeline([
    ('vec', make_vectorizer()),
    ('rf', RandomForestClassifier(n_estimators=params['random_forest']['n_estimators'],
                                  max_features=params['random_forest']['max_features'],
                                  random_state=seed, n_jobs=-1)),
])

gb = Pipeline([
    ('vec', make_vectorizer()),
    ('gb', GradientBoostingClassifier(random_state=seed)),
])
gb.set_params(**params['gradient_boosting'])

models = {
    'naive_bayes': nb,
    'logistic_regression': lr,
    'decision_tree': tree,
    'random_forest': rf,
    'gradient_boosting': gb,
}


### Train on the full training set, evaluate on the test set ###
results = []

for name, model in models.items():
    print('Training', name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, pos_label=positive_label),
        'recall': recall_score(y_test, y_pred, pos_label=positive_label),
        'f1': f1_score(y_test, y_pred, pos_label=positive_label),
    })

results_df = pd.DataFrame(results)
print(results_df)

results_df.to_csv('test_results.csv', index=False)
