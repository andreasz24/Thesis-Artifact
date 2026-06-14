"""
Model training functions — one section per model.

Heavy / optional dependencies (xgboost, tabpfn, autogluon) are imported
*inside* the functions that need them, so importing this module never fails
just because one library isn't installed. A notebook that only runs XGBoost
doesn't need TabPFN or AutoGluon present.

The model configurations are copied verbatim from the original Colab script.
"""

import time

import joblib
import numpy as np
import pandas as pd

from src import config
from src import evaluation


# ══════════════════════════════════════════════════════════════════════════
# XGBOOST  (tuned gradient-boosting baseline)
# ══════════════════════════════════════════════════════════════════════════
def train_xgboost(X_train, y_train, X_test, y_test):
    """Fit the XGBoost baseline. Returns (model, train_time_seconds)."""
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators      = 500,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        use_label_encoder = False,
        eval_metric       = "mlogloss",
        random_state      = config.RANDOM_STATE,
        n_jobs            = -1,
    )

    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    train_time = time.time() - t0
    return model, train_time


def evaluate_and_save_xgboost(model, X_test, y_test, train_time, save=True):
    """Predict, compute metrics, and save results in the shared format."""
    t1 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t1

    metrics = evaluation.compute_metrics(y_test, y_pred)
    results = {
        "model":            "XGBoost",
        "accuracy":         metrics["accuracy"],
        "f1_macro":         metrics["f1_macro"],
        "f1_weighted":      metrics["f1_weighted"],
        "train_time_s":     train_time,
        "inference_time_s": inference_time,
        "y_pred":           y_pred,
    }
    if save:
        joblib.dump(results, config.XGB_RESULTS_PATH)
    return results, y_pred, inference_time


# ══════════════════════════════════════════════════════════════════════════
# TABPFN-3  (tabular foundation model)
# ══════════════════════════════════════════════════════════════════════════
def get_device():
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_tabpfn(n_estimators, device):
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(
        n_estimators              = n_estimators,
        device                    = device,
        ignore_pretraining_limits = True,
        random_state              = config.RANDOM_STATE,
    )


def run_tabpfn_sweep(X_train, y_train, X_test, y_test,
                     n_estimators_range=None, device=None, verbose=True):
    """Sweep n_estimators and return a results DataFrame.

    Each estimator is a separate forward pass through the transformer using a
    different random configuration (e.g. row ordering); predictions are
    averaged. Mirrors the original experiment loop exactly.
    """
    n_estimators_range = n_estimators_range or config.N_ESTIMATORS_RANGE
    device = device or get_device()

    rows = []
    for n_est in n_estimators_range:
        if verbose:
            print(f"\n{'-'*50}\n  n_estimators = {n_est}\n{'-'*50}")

        clf = _make_tabpfn(n_est, device)

        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - t0

        t1 = time.time()
        y_pred = clf.predict(X_test)
        inference_time = time.time() - t1

        m = evaluation.compute_metrics(y_test, y_pred)
        rows.append({
            "n_estimators":     n_est,
            "accuracy":         round(m["accuracy"], 4),
            "f1_macro":         round(m["f1_macro"], 4),
            "f1_weighted":      round(m["f1_weighted"], 4),
            "fit_time_s":       round(fit_time, 2),
            "inference_time_s": round(inference_time, 2),
            "total_time_s":     round(fit_time + inference_time, 2),
        })
        if verbose:
            print(f"  Accuracy   : {m['accuracy']:.4f}")
            print(f"  F1 Macro   : {m['f1_macro']:.4f}")
            print(f"  Fit time   : {fit_time:.2f}s | Infer: {inference_time:.2f}s")

    return pd.DataFrame(rows)


def train_tabpfn_best(X_train, y_train, X_test, y_test, best_n,
                      device=None, save=True):
    """Retrain TabPFN-3 at the chosen n_estimators for the final evaluation."""
    device = device or get_device()
    clf = _make_tabpfn(best_n, device)

    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - t0

    t1 = time.time()
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    inference_time = time.time() - t1

    m = evaluation.compute_metrics(y_test, y_pred)
    results = {
        "model":            "TabPFN-3",
        "accuracy":         m["accuracy"],
        "f1_macro":         m["f1_macro"],
        "f1_weighted":      m["f1_weighted"],
        "fit_time_s":       fit_time,
        "inference_time_s": inference_time,
        "n_estimators":     best_n,
        "y_pred":           y_pred,
        "y_prob":           y_prob,
    }
    if save:
        joblib.dump(results, config.TABPFN_RESULTS_PATH)
    return clf, results, y_pred


# ══════════════════════════════════════════════════════════════════════════
# AUTOGLUON  (AutoML framework)
# ══════════════════════════════════════════════════════════════════════════
def _to_tabular(X, y):
    """AutoGluon needs the label inside the dataframe."""
    data = X.copy()
    data[config.TARGET_COL] = y.values
    return data


def run_autogluon_grid(X_train, y_train, X_test, y_test,
                       folds=None, stacks=None, time_limit=None, verbose=True):
    """2x2 grid over num_bag_folds x num_stack_levels.

    Selection metric is AutoGluon's internal CV score (score_val) on the
    training set; the test set is also scored for reporting. Returns a
    results DataFrame.
    """
    from autogluon.tabular import TabularPredictor
    from sklearn.metrics import accuracy_score, f1_score

    folds      = folds or config.FOLDS
    stacks     = stacks or config.STACKS
    time_limit = time_limit or config.TIME_LIMIT

    train_data = _to_tabular(X_train, y_train)
    test_data  = _to_tabular(X_test, y_test)

    rows = []
    for num_folds in folds:
        for num_stacks in stacks:
            config_name = f"folds{num_folds}_stacks{num_stacks}"
            model_path  = str(config.AG_GRID_DIR / config_name)

            if verbose:
                print(f"\n{'='*60}\n  num_bag_folds={num_folds}, "
                      f"num_stack_levels={num_stacks}\n{'='*60}")

            t0 = time.time()
            predictor = TabularPredictor(
                label        = config.TARGET_COL,
                problem_type = "multiclass",
                eval_metric  = "f1_macro",
                path         = model_path,
                verbosity    = 0,
            ).fit(
                train_data       = train_data,
                time_limit       = time_limit,
                presets          = "medium_quality",
                num_bag_folds    = num_folds,
                num_stack_levels = num_stacks,
            )
            train_time = time.time() - t0

            best_val = predictor.leaderboard(silent=True)["score_val"].max()

            y_pred = predictor.predict(
                test_data.drop(columns=[config.TARGET_COL])
            ).astype(int)
            acc    = accuracy_score(y_test, y_pred)
            f1_mac = f1_score(y_test, y_pred, average="macro",    zero_division=0)
            f1_wtd = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            rows.append({
                "num_bag_folds":    num_folds,
                "num_stack_levels": num_stacks,
                "cv_f1_macro":      round(best_val, 4),
                "test_accuracy":    round(acc, 4),
                "test_f1_macro":    round(f1_mac, 4),
                "test_f1_weighted": round(f1_wtd, 4),
                "train_time_s":     round(train_time, 1),
            })
            if verbose:
                print(f"  CV F1 Macro: {best_val:.4f} | "
                      f"Test F1 Macro: {f1_mac:.4f} | {train_time:.1f}s")

    return pd.DataFrame(rows)


def train_autogluon_best(X_train, y_train, X_test, y_test,
                         best_folds, best_stacks, time_limit=None, save=True):
    """Retrain the best AutoGluon config and save results in the shared format."""
    from autogluon.tabular import TabularPredictor
    from sklearn.metrics import accuracy_score, f1_score

    time_limit = time_limit or config.TIME_LIMIT
    train_data = _to_tabular(X_train, y_train)
    test_data  = _to_tabular(X_test, y_test)

    t0 = time.time()
    predictor = TabularPredictor(
        label        = config.TARGET_COL,
        problem_type = "multiclass",
        eval_metric  = "f1_macro",
        path         = str(config.AG_BEST_DIR),
        verbosity    = 1,
    ).fit(
        train_data       = train_data,
        time_limit       = time_limit,
        presets          = "medium_quality",
        num_bag_folds    = best_folds,
        num_stack_levels = best_stacks,
    )
    train_time = time.time() - t0

    y_pred = predictor.predict(
        test_data.drop(columns=[config.TARGET_COL])
    ).astype(int)
    m = evaluation.compute_metrics(y_test, y_pred)
    results = {
        "model":            "AutoGluon",
        "accuracy":         m["accuracy"],
        "f1_macro":         m["f1_macro"],
        "f1_weighted":      m["f1_weighted"],
        "train_time_s":     train_time,
        "num_bag_folds":    best_folds,
        "num_stack_levels": best_stacks,
        "y_pred":           y_pred,
    }
    if save:
        joblib.dump(results, config.AUTOGLUON_RESULTS_PATH)
    return predictor, results, y_pred
