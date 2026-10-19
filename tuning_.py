import pandas as pd
from tuning_functions import tune_naive_bayes, tune_logistic_regression, tune_single_tree,tune_random_forests, tune_gradient_boosting
import joblib

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df['text'] = train_df['text'].fillna('')
test_df['text'] = test_df['text'].fillna('')

X_train, y_train = train_df['text'], train_df['source']
X_test, y_test = test_df['text'], test_df['source']

tuned_nb = tune_naive_bayes(X_train, y_train)
tuned_lr = tune_logistic_regression(X_train, y_train)
tuned_rf = tune_random_forests(X_train, y_train)
tuned_tree = tune_single_tree(X_train, y_train)
tuned_gb = tune_gradient_boosting(X_train, y_train)

models = {
    'naive_bayes': tuned_nb,
    'logistic_regression': tuned_lr,
    'random_forest': tuned_rf,
    'decision_tree': tuned_tree,
    'gradient_boosting': tuned_gb,
}

joblib.dump(models, 'tuned_models.joblib')
