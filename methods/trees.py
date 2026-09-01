import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def build_rf_estimator(task_type, seed=42, n_train=0):
    min_leaf = max(5, int(np.ceil(0.03 * max(n_train, 1))))
    min_split = max(12, int(np.ceil(0.10 * max(n_train, 1))), 2 * min_leaf)

    common = dict(
        n_estimators=50,
        random_state=seed,
        n_jobs=-1,
        max_depth=10,
        min_samples_leaf=min_leaf,
        min_samples_split=min_split,
        max_features=0.15,  # 15% of the retained ECFP bits per split.
        bootstrap=True,
        max_samples=0.15,
    )

    if task_type in (1, 2):
        return RandomForestClassifier(**common)
    return RandomForestRegressor(**common)
