from sklearn.model_selection import StratifiedKFold

seed = 42
mindf = 5
token = r'\b[^\W\d_]{2,}\b'

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
