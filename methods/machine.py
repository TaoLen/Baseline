import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR


def build_svm_estimator(
    task_type,
    seed=42,
    n_features=2048,
    c_value=0.01,
    epsilon=0.2,
):
    max_selected = max(32, int(np.ceil(0.30 * max(n_features, 1))))

    if task_type in (1, 2):
        selector = SelectFromModel(
            estimator=LinearSVC(
                C=0.01,
                penalty="l1",
                dual=False,
                max_iter=10000,
                random_state=seed,
            ),
            threshold="median",
            max_features=max_selected,
        )
        classifier = SVC(
            C=c_value,
            gamma="scale",
            kernel="poly",
            probability=True,
            random_state=seed,
        )
        return make_pipeline(
            StandardScaler(),
            selector,
            classifier,
        )

    selector = SelectFromModel(
        estimator=LinearSVR(
            C=0.01,
            epsilon=epsilon,
            max_iter=10000,
            random_state=seed,
        ),
        threshold="median",
        max_features=max_selected,
    )
    regressor = SVR(
        C=c_value,
        gamma="scale",
        kernel="poly",
        epsilon=epsilon,
    )
    return make_pipeline(
        StandardScaler(),
        selector,
        regressor,
    )


def get_selected_feature_count(model):
    if hasattr(model, "named_steps") and "selectfrommodel" in model.named_steps:
        mask = model.named_steps["selectfrommodel"].get_support()
        return int(mask.sum())
    return None

