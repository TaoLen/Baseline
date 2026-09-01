import json
import os

import joblib
import numpy as np
import torch
from rdkit import Chem
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from fingerprints import calc_fp
from machine import build_svm_estimator, get_selected_feature_count
from metrics import ClassificationMetrics, MulticlassMetrics, RegressionMetrics
from save import save_thresholds
from statistical import (
    apply_thresholds,
    best_thresholds,
    classification_markdown,
    multiclass_markdown,
    regression_markdown,
    standard_threshold,
    threshold_moving,
)
from trees import build_rf_estimator
from utils import task_inference


def smiles_to_ecfp(smiles, fp_size=2048, radius=2):
    X = np.zeros((len(smiles), fp_size), dtype=np.float32)
    invalid = []
    for i, smi in enumerate(smiles):
        if Chem.MolFromSmiles(smi, sanitize=True) is None:
            invalid.append(i)
            continue
        X[i] = calc_fp(smi, fp_size=fp_size, radius=radius)
    return X, np.asarray(invalid, dtype=int)


def fit_zero_variance_filter(X_train, X_val, X_test):
    """Fit a zero-variance bit filter on train and apply it to every split.

    The fitted ``VarianceThreshold`` is returned so the exact same bit mask can
    be serialized with each estimator and reused for future inference.
    """
    splits = {
        "train": np.asarray(X_train, dtype=np.float32),
        "validation": np.asarray(X_val, dtype=np.float32),
        "test": np.asarray(X_test, dtype=np.float32),
    }

    for split_name, X in splits.items():
        if X.ndim != 2:
            raise ValueError(
                f"X_{split_name} must be a 2D matrix; got shape {X.shape}."
            )

    n_features = splits["train"].shape[1]
    if splits["train"].shape[0] == 0:
        raise ValueError("Cannot fit the fingerprint filter on an empty train set.")
    for split_name in ("validation", "test"):
        if splits[split_name].shape[1] != n_features:
            raise ValueError(
                "All fingerprint matrices must have the same number of bits; "
                f"train has {n_features}, but {split_name} has "
                f"{splits[split_name].shape[1]}."
            )

    bit_filter = VarianceThreshold(threshold=0.0)
    X_train_filtered = bit_filter.fit_transform(splits["train"]).astype(
        np.float32, copy=False
    )
    X_val_filtered = bit_filter.transform(splits["validation"]).astype(
        np.float32, copy=False
    )
    X_test_filtered = bit_filter.transform(splits["test"]).astype(
        np.float32, copy=False
    )

    return X_train_filtered, X_val_filtered, X_test_filtered, bit_filter


def infer_task_metadata_from_train(y_train):
    y_train = np.asarray(y_train, dtype=np.float32)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)

    mask = ~np.isnan(y_train)
    y_fill = np.nan_to_num(y_train, nan=0.0)
    task_type = task_inference(
        torch.tensor(y_fill, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.bool),
    ).cpu().numpy()

    mc_label_values = [None] * y_train.shape[1]
    for j in range(y_train.shape[1]):
        if task_type[j] == 2:
            vals = y_train[mask[:, j], j]
            labels = sorted(np.unique(np.rint(vals).astype(int)).tolist())
            mc_label_values[j] = labels

    return task_type, mc_label_values


def _get_classes(model):
    if hasattr(model, "classes_"):
        return model.classes_
    if hasattr(model, "steps"):
        return model.steps[-1][1].classes_
    return None


def _binary_positive_proba(model, X):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        classes = _get_classes(model)
        if classes is None:
            return probs[:, -1].astype(np.float32)
        cls_arr = np.asarray(classes)
        if 1 in cls_arr:
            pos_col = int(np.where(cls_arr == 1)[0][0])
            return probs[:, pos_col].astype(np.float32)
        return probs[:, -1].astype(np.float32)

    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        return (1.0 / (1.0 + np.exp(-score))).astype(np.float32)

    preds = model.predict(X)
    return np.asarray(preds, dtype=np.float32)


def _multiclass_proba(model, X, num_classes):
    probs = model.predict_proba(X)
    classes = _get_classes(model)
    if classes is None:
        if probs.shape[1] != num_classes:
            raise ValueError("Multiclass probabilities have unexpected shape.")
        return probs.astype(np.float32)

    full = np.zeros((X.shape[0], num_classes), dtype=np.float32)
    for local_col, cls in enumerate(np.asarray(classes).tolist()):
        idx = int(cls)
        if idx < 0 or idx >= num_classes:
            raise ValueError("Unexpected multiclass class index in model.")
        full[:, idx] = probs[:, local_col].astype(np.float32)
    return full


def _build_model(method, task_type, seed, n_train, n_features, c_value, epsilon):
    if method == "rf":
        return build_rf_estimator(
            task_type=task_type,
            seed=seed,
            n_train=n_train,
        )
    if method == "svm":
        return build_svm_estimator(
            task_type=task_type,
            seed=seed,
            n_features=n_features,
            c_value=c_value,
            epsilon=epsilon,
        )
    raise ValueError(f"Unknown method: {method}")


def fit_predict_multitask_cv(
    method,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    task_type,
    mc_label_values,
    model_out_dir,
    params_out_dir,
    n_splits=5,
    seed=42,
    c_value=0.01,
    epsilon=0.2,
):
    os.makedirs(model_out_dir, exist_ok=True)
    os.makedirs(params_out_dir, exist_ok=True)

    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    original_n_features = X_train.shape[1] if X_train.ndim == 2 else None
    X_train, X_val, X_test, bit_filter = fit_zero_variance_filter(
        X_train,
        X_val,
        X_test,
    )
    retained_bit_indices = np.flatnonzero(bit_filter.get_support()).astype(int)
    removed_bit_indices = np.flatnonzero(~bit_filter.get_support()).astype(int)

    bit_filter_file = os.path.join(
        params_out_dir,
        f"{method}_zero_variance_filter.joblib",
    )
    joblib.dump(bit_filter, bit_filter_file)

    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    n_trainval, n_tasks = y_trainval.shape
    n_test = y_test.shape[0]

    pred_train = np.full((n_trainval, n_tasks), np.nan, dtype=np.float32)
    pred_val = np.full((n_trainval, n_tasks), np.nan, dtype=np.float32)
    pred_test = np.full((n_test, n_tasks), np.nan, dtype=np.float32)

    mc_probs_train = {}
    mc_probs_val = {}
    mc_probs_test = {}

    model_info = {
        "method": method,
        "cv_type": f"{n_splits}-fold",
        "trainval_samples": int(n_trainval),
        "fingerprint_filter": {
            "type": "VarianceThreshold",
            "threshold": 0.0,
            "fit_split": "train",
            "index_base": 0,
            "original_bits": int(original_n_features),
            "retained_bits": int(retained_bit_indices.size),
            "removed_bits": int(removed_bit_indices.size),
            "retained_bit_indices": retained_bit_indices.tolist(),
            "removed_bit_indices": removed_bit_indices.tolist(),
            "artifact": bit_filter_file,
        },
        "tasks": [],
    }

    for j in range(n_tasks):
        ttype = int(task_type[j])
        y_task = y_trainval[:, j]
        valid_idx = np.where(~np.isnan(y_task))[0]

        info = {
            "task": j + 1,
            "task_type": ttype,
            "trainval_valid_samples": int(valid_idx.size),
            "fold_models": [],
        }

        if valid_idx.size < 2:
            info["status"] = "skipped_not_enough_valid_labels"
            model_info["tasks"].append(info)
            continue

        label_values = None
        label_to_idx = None
        y_labels = None
        if ttype == 2:
            label_values = mc_label_values[j]
            if label_values is None:
                label_values = sorted(
                    np.unique(np.rint(y_task[valid_idx]).astype(int)).tolist()
                )
            if len(label_values) < 3:
                info["status"] = "skipped_invalid_multiclass_labels"
                model_info["tasks"].append(info)
                continue
            label_to_idx = {int(v): i for i, v in enumerate(label_values)}
            y_labels_raw = np.rint(y_task[valid_idx]).astype(int)
            try:
                y_labels = np.asarray(
                    [label_to_idx[int(v)] for v in y_labels_raw],
                    dtype=int,
                )
            except KeyError:
                info["status"] = "skipped_unseen_multiclass_label"
                model_info["tasks"].append(info)
                continue
        elif ttype == 1:
            y_labels = np.rint(y_task[valid_idx]).astype(int)

        if ttype in (1, 2):
            unique_labels, counts = np.unique(y_labels, return_counts=True)
            if unique_labels.size < 2:
                info["status"] = "skipped_single_class_trainval"
                model_info["tasks"].append(info)
                continue
            n_splits_task = min(int(n_splits), int(counts.min()))
            if n_splits_task < 2:
                info["status"] = "skipped_insufficient_class_support_for_cv"
                model_info["tasks"].append(info)
                continue
            splitter = StratifiedKFold(
                n_splits=n_splits_task,
                shuffle=True,
                random_state=seed + j,
            )
            splits = splitter.split(valid_idx, y_labels)
        else:
            n_splits_task = min(int(n_splits), int(valid_idx.size))
            if n_splits_task < 2:
                info["status"] = "skipped_insufficient_samples_for_cv"
                model_info["tasks"].append(info)
                continue
            splitter = KFold(
                n_splits=n_splits_task,
                shuffle=True,
                random_state=seed + j,
            )
            splits = splitter.split(valid_idx)

        train_sum = np.zeros(n_trainval, dtype=np.float64)
        train_count = np.zeros(n_trainval, dtype=np.float64)
        val_oof = np.full(n_trainval, np.nan, dtype=np.float32)
        test_sum = np.zeros(n_test, dtype=np.float64)
        test_count = 0

        train_prob_sum = None
        train_prob_count = None
        val_prob_oof = None
        test_prob_sum = None
        if ttype == 2:
            num_classes = len(label_values)
            train_prob_sum = np.zeros((n_trainval, num_classes), dtype=np.float64)
            train_prob_count = np.zeros(n_trainval, dtype=np.float64)
            val_prob_oof = np.full((n_trainval, num_classes), np.nan, dtype=np.float32)
            test_prob_sum = np.zeros((n_test, num_classes), dtype=np.float64)

        selected_features = []
        fold_ok = 0

        for fold_id, (tr_rel, va_rel) in enumerate(splits, start=1):
            tr_idx = valid_idx[tr_rel]
            va_idx = valid_idx[va_rel]
            y_fit_raw = y_task[tr_idx]

            if ttype == 0:
                y_fit = y_fit_raw.astype(np.float32)
            elif ttype == 1:
                y_fit = np.rint(y_fit_raw).astype(int)
                if np.unique(y_fit).size < 2:
                    continue
            else:
                y_fit_int = np.rint(y_fit_raw).astype(int)
                y_fit = np.asarray(
                    [label_to_idx[int(v)] for v in y_fit_int],
                    dtype=int,
                )
                if np.unique(y_fit).size < 2:
                    continue

            model = _build_model(
                method=method,
                task_type=ttype,
                seed=seed + 1000 * j + fold_id,
                n_train=int(tr_idx.size),
                n_features=X_trainval.shape[1],
                c_value=c_value,
                epsilon=epsilon,
            )
            model.fit(X_trainval[tr_idx], y_fit)

            fold_model_file = os.path.join(
                model_out_dir,
                f"{method}_task_{j+1}_fold_{fold_id}.joblib",
            )
            serialized_model = Pipeline(
                [
                    ("zero_variance_filter", bit_filter),
                    ("estimator", model),
                ]
            )
            joblib.dump(serialized_model, fold_model_file)
            info["fold_models"].append(fold_model_file)

            if method == "svm":
                n_sel = get_selected_feature_count(model)
                if n_sel is not None:
                    selected_features.append(int(n_sel))

            if ttype == 0:
                p_train_fold = model.predict(X_trainval[tr_idx]).astype(np.float32)
                p_val_fold = model.predict(X_trainval[va_idx]).astype(np.float32)
                p_test_fold = model.predict(X_test).astype(np.float32)

                train_sum[tr_idx] += p_train_fold
                train_count[tr_idx] += 1.0
                val_oof[va_idx] = p_val_fold
                test_sum += p_test_fold
            elif ttype == 1:
                p_train_fold = _binary_positive_proba(model, X_trainval[tr_idx])
                p_val_fold = _binary_positive_proba(model, X_trainval[va_idx])
                p_test_fold = _binary_positive_proba(model, X_test)

                train_sum[tr_idx] += p_train_fold
                train_count[tr_idx] += 1.0
                val_oof[va_idx] = p_val_fold
                test_sum += p_test_fold
            else:
                p_train_fold = _multiclass_proba(
                    model,
                    X_trainval[tr_idx],
                    num_classes=len(label_values),
                )
                p_val_fold = _multiclass_proba(
                    model,
                    X_trainval[va_idx],
                    num_classes=len(label_values),
                )
                p_test_fold = _multiclass_proba(
                    model,
                    X_test,
                    num_classes=len(label_values),
                )

                train_prob_sum[tr_idx] += p_train_fold
                train_prob_count[tr_idx] += 1.0
                val_prob_oof[va_idx] = p_val_fold
                test_prob_sum += p_test_fold

            fold_ok += 1
            test_count += 1

        if fold_ok == 0:
            info["status"] = "skipped_all_folds_invalid"
            model_info["tasks"].append(info)
            continue

        if method == "svm" and selected_features:
            info["selected_features_mean"] = float(np.mean(selected_features))
            info["selected_features_min"] = int(np.min(selected_features))
            info["selected_features_max"] = int(np.max(selected_features))

        if ttype in (0, 1):
            m_train = train_count > 0
            pred_train[m_train, j] = (train_sum[m_train] / train_count[m_train]).astype(
                np.float32
            )
            pred_val[:, j] = val_oof
            if test_count > 0:
                pred_test[:, j] = (test_sum / float(test_count)).astype(np.float32)
        else:
            m_train = train_prob_count > 0
            train_prob_avg = np.full_like(train_prob_sum, np.nan, dtype=np.float32)
            train_prob_avg[m_train] = (
                train_prob_sum[m_train] / train_prob_count[m_train, None]
            ).astype(np.float32)
            test_prob_avg = (test_prob_sum / float(test_count)).astype(np.float32)

            mc_probs_train[j] = train_prob_avg
            mc_probs_val[j] = val_prob_oof
            mc_probs_test[j] = test_prob_avg

            idx_to_label = np.asarray(label_values, dtype=int)
            if m_train.any():
                pred_train[m_train, j] = idx_to_label[
                    train_prob_avg[m_train].argmax(axis=1)
                ].astype(np.float32)

            m_val = ~np.isnan(val_prob_oof).all(axis=1)
            if m_val.any():
                pred_val[m_val, j] = idx_to_label[
                    val_prob_oof[m_val].argmax(axis=1)
                ].astype(np.float32)

            pred_test[:, j] = idx_to_label[
                test_prob_avg.argmax(axis=1)
            ].astype(np.float32)

        info["n_folds_used"] = int(fold_ok)
        info["status"] = "trained"
        model_info["tasks"].append(info)

    params_file = os.path.join(params_out_dir, f"{method}_cv_metadata.json")
    with open(params_file, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4)

    return {
        "y_trainval": y_trainval,
        "pred_train": pred_train,
        "pred_val": pred_val,
        "pred_test": pred_test,
        "mc_probs_train": mc_probs_train,
        "mc_probs_val": mc_probs_val,
        "mc_probs_test": mc_probs_test,
        "metadata": model_info,
    }


def evaluate_ml_predictions_cv(
    method_name,
    task_type,
    y_true_trainval,
    y_true_test,
    y_pred_train,
    y_pred_val,
    y_pred_test,
    y_prob_train_mc,
    y_prob_val_mc,
    y_prob_test_mc,
    mc_label_values,
    calibration=False,
):
    cls_calc = ClassificationMetrics(None, None)
    reg_calc = RegressionMetrics(None, None)
    mc_calc = MulticlassMetrics()

    idx_bin = np.where(task_type == 1)[0].tolist()
    idx_mc = np.where(task_type == 2)[0].tolist()
    idx_reg = np.where(task_type == 0)[0].tolist()

    def slice_cols(arr, cols):
        if arr is None or len(cols) == 0:
            return None
        return arr[:, cols]

    print(f"\n### {method_name.upper()} (5-fold CV train+val, hold-out test)")

    if idx_bin:
        ypt = slice_cols(y_pred_train, idx_bin)
        ypv = slice_cols(y_pred_val, idx_bin)
        yps = slice_cols(y_pred_test, idx_bin)

        ytt = slice_cols(y_true_trainval, idx_bin)
        ytv = slice_cols(y_true_trainval, idx_bin)
        yts = slice_cols(y_true_test, idx_bin)

        if calibration:
            train_results, val_results = threshold_moving(
                cls_calc, ypt, ytt, ypv, ytv
            )
            train_metrics, val_metrics, avg_thresholds = best_thresholds(
                train_results, val_results, ytt.shape[1]
            )
            test_metrics = apply_thresholds(yps, yts, avg_thresholds["val"], cls_calc)
            avg_thresholds["test"] = avg_thresholds["val"]
        else:
            default_thresh = 0.5
            train_metrics, val_metrics, test_metrics = standard_threshold(
                ypt, ytt, ypv, ytv, yps, yts, default_thresh, cls_calc
            )
            avg_thresholds = {
                "train": [default_thresh] * ytt.shape[1],
                "val": [default_thresh] * ytv.shape[1],
                "test": [default_thresh] * yts.shape[1],
            }

        save_thresholds(
            avg_thresholds,
            out_path=f"../output/calibration/thresholds_{method_name}_cv5.json",
        )

        def flat_valid(pred, true):
            mk = ~np.isnan(true)
            return pred[mk].reshape(-1, 1), true[mk].reshape(-1, 1)

        trp_g, trt_g = flat_valid(ypt, ytt)
        vap_g, vat_g = flat_valid(ypv, ytv)
        tep_g, tet_g = flat_valid(yps, yts)

        tr_avg = cls_calc.calculate_metrics(
            trp_g, trt_g, float(np.mean(avg_thresholds["train"]))
        )[0]
        va_avg = cls_calc.calculate_metrics(
            vap_g, vat_g, float(np.mean(avg_thresholds["val"]))
        )[0]
        te_avg = cls_calc.calculate_metrics(
            tep_g, tet_g, float(np.mean(avg_thresholds["test"]))
        )[0]

        classification_markdown(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            global_train_metrics=tr_avg,
            global_val_metrics=va_avg,
            global_test_metrics=te_avg,
            avg_train_threshold=float(np.mean(avg_thresholds["train"])),
            avg_val_threshold=float(np.mean(avg_thresholds["val"])),
            avg_test_threshold=float(np.mean(avg_thresholds["test"])),
            task_indices=[i + 1 for i in idx_bin],
        )
    else:
        print("No binary classification tasks detected.")

    if idx_mc:
        train_metrics = []
        val_metrics = []
        test_metrics = []

        for task_idx in idx_mc:
            probs_tr = y_prob_train_mc.get(task_idx)
            probs_va = y_prob_val_mc.get(task_idx)
            probs_te = y_prob_test_mc.get(task_idx)
            if probs_te is None:
                continue

            def slice_task(probs, y_true):
                true = y_true[:, task_idx]
                mask = ~np.isnan(true)
                if not mask.any():
                    return None, None
                labels = np.rint(true[mask]).astype(int)
                mapping_vals = mc_label_values[task_idx]
                mapping = {int(v): i for i, v in enumerate(mapping_vals)}
                mapped = np.asarray([mapping[int(v)] for v in labels.tolist()], dtype=int)
                return probs[mask], mapped

            pte, yte = slice_task(probs_te, y_true_test)
            if pte is None:
                continue

            if probs_tr is not None:
                ptr, ytr = slice_task(probs_tr, y_true_trainval)
                train_metrics.append(mc_calc.calculate_metrics(ptr, ytr))
            else:
                train_metrics.append(mc_calc.calculate_metrics(pte, yte))

            if probs_va is not None:
                pva, yva = slice_task(probs_va, y_true_trainval)
                val_metrics.append(mc_calc.calculate_metrics(pva, yva))
            else:
                val_metrics.append(mc_calc.calculate_metrics(pte, yte))

            test_metrics.append(mc_calc.calculate_metrics(pte, yte))

        if test_metrics:
            def avg_metric(metrics_list, key):
                vals = [m[key] for m in metrics_list]
                vals = [v for v in vals if np.isfinite(v)]
                return float(np.mean(vals)) if vals else np.nan

            global_train_metrics = {
                "accuracy": avg_metric(train_metrics, "accuracy"),
                "balanced_accuracy_macro": avg_metric(
                    train_metrics, "balanced_accuracy_macro"
                ),
                "f1_macro": avg_metric(train_metrics, "f1_macro"),
                "mcc": avg_metric(train_metrics, "mcc"),
                "prauc_macro": avg_metric(train_metrics, "prauc_macro"),
                "auc_ovr": avg_metric(train_metrics, "auc_ovr"),
            }
            global_val_metrics = {
                "accuracy": avg_metric(val_metrics, "accuracy"),
                "balanced_accuracy_macro": avg_metric(
                    val_metrics, "balanced_accuracy_macro"
                ),
                "f1_macro": avg_metric(val_metrics, "f1_macro"),
                "mcc": avg_metric(val_metrics, "mcc"),
                "prauc_macro": avg_metric(val_metrics, "prauc_macro"),
                "auc_ovr": avg_metric(val_metrics, "auc_ovr"),
            }
            global_test_metrics = {
                "accuracy": avg_metric(test_metrics, "accuracy"),
                "balanced_accuracy_macro": avg_metric(
                    test_metrics, "balanced_accuracy_macro"
                ),
                "f1_macro": avg_metric(test_metrics, "f1_macro"),
                "mcc": avg_metric(test_metrics, "mcc"),
                "prauc_macro": avg_metric(test_metrics, "prauc_macro"),
                "auc_ovr": avg_metric(test_metrics, "auc_ovr"),
            }

            multiclass_markdown(
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                global_train_metrics=global_train_metrics,
                global_val_metrics=global_val_metrics,
                global_test_metrics=global_test_metrics,
                task_indices=[i + 1 for i in idx_mc],
            )
        else:
            print("No multiclass tasks detected.")
    else:
        print("No multiclass tasks detected.")

    if idx_reg:
        ypt = slice_cols(y_pred_train, idx_reg)
        ypv = slice_cols(y_pred_val, idx_reg)
        yps = slice_cols(y_pred_test, idx_reg)
        ytt = slice_cols(y_true_trainval, idx_reg)
        ytv = slice_cols(y_true_trainval, idx_reg)
        yts = slice_cols(y_true_test, idx_reg)

        tr = reg_calc.calculate_metrics(ypt, ytt)
        va = reg_calc.calculate_metrics(ypv, ytv)
        te = reg_calc.calculate_metrics(yps, yts)

        def flatten_valid(pred, true):
            mk = ~np.isnan(true)
            return pred[mk].reshape(-1, 1), true[mk].reshape(-1, 1)

        trp, trt = flatten_valid(ypt, ytt)
        vap, vat = flatten_valid(ypv, ytv)
        tep, tet = flatten_valid(yps, yts)

        tr_g = reg_calc.calculate_metrics(trp, trt)[0]
        va_g = reg_calc.calculate_metrics(vap, vat)[0]
        te_g = reg_calc.calculate_metrics(tep, tet)[0]

        regression_markdown(
            train_metrics=tr,
            val_metrics=va,
            test_metrics=te,
            global_train_metrics=tr_g,
            global_val_metrics=va_g,
            global_test_metrics=te_g,
            task_indices=[i + 1 for i in idx_reg],
        )
    else:
        print("No regression tasks detected.")
