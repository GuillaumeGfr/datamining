import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df['text'] = train_df['text'].fillna('')
test_df['text'] = test_df['text'].fillna('')

X_train, y_train = train_df['text'], train_df['source']
X_test, y_test = test_df['text'], test_df['source']

models = joblib.load('tuned_models.joblib')

results = []

for name, m in models.items():
    y_pred = m['model'].predict(X_test)

    results.append({
        'model': name,
        'params': m['params'],
        'cv_score': m['cv_score'],
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred, pos_label=1),
        'test_recall': recall_score(y_test, y_pred, pos_label=1),
        'test_f1': f1_score(y_test, y_pred, pos_label=1),
    })

results_df = pd.DataFrame(results)
results_df.to_csv('test_results.csv', index=False)
