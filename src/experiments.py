"""Reproducible multi-seed experimental runners and cross-validation utilities."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.datasets import (
    generate_synthetic_concentric_rings,
    load_uci_digits_even_odd,
    prepare_dataset_zero_leakage,
    standardize_features_zero_leakage,
    stratified_kfold_indices,
)
from src.kernels import (
    compute_laplacian_gram_matrix,
    compute_rbf_gram_matrix,
    kernel_matrix_approximation_errors,
)
from src.models import (
    KernelPegasosSVM,
    PrimalPegasosSVM,
)
from src.random_features import (
    OrthogonalRandomFeatures,
    RandomBinningFeatures,
    RFFCosinePhase,
    RFFCosSinPair,
)


def compute_classification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the empirical 0-1 classification error rate in [0.0, 1.0].

    Parameters
    ----------
    y_true : np.ndarray of shape (m,)
        Ground-truth binary class labels in {-1.0, +1.0}.
    y_pred : np.ndarray of shape (m,)
        Predicted binary class labels in {-1.0, +1.0}.

    Returns
    -------
    error_rate : float
        Fraction of misclassified examples mean(y_true != y_pred).
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError(
            f"y_true and y_pred must be 1D arrays, got shapes {y_true_arr.shape} and {y_pred_arr.shape}."
        )
    if y_true_arr.shape[0] == 0 or y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have matching non-zero length, got {y_true_arr.shape[0]} vs {y_pred_arr.shape[0]}."
        )
    return float(np.mean(y_true_arr != y_pred_arr))


def _fit_loglog_slope(x_vals: np.ndarray, y_vals: np.ndarray) -> float:
    """
    Estimate the power-law exponent alpha in y ~ c * x^alpha via ordinary least squares on log(x) and log(y).

    Parameters
    ----------
    x_vals : np.ndarray of shape (n,)
        Strictly positive independent variable values (e.g. D_grid).
    y_vals : np.ndarray of shape (n,)
        Strictly positive dependent metric values (e.g. mean Gram error).

    Returns
    -------
    slope : float
        Least-squares slope d(log y) / d(log x).
    """
    log_x = np.log(np.asarray(x_vals, dtype=np.float64))
    log_y = np.log(np.maximum(np.asarray(y_vals, dtype=np.float64), 1e-15))
    x_centered = log_x - np.mean(log_x)
    y_centered = log_y - np.mean(log_y)
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denom)


def cross_validate_exact_kernel(
    X_train: np.ndarray,
    y_train: np.ndarray,
    gamma_grid: Sequence[float],
    lambda_grid: Sequence[float],
    model_type: str = "pegasos",
    n_splits: int = 5,
    standardize_within_folds: bool = False,
    pegasos_iters: int = 15000,
    seed: int = 42,
    kernel_type: str = "rbf",
    selection_rule: str = "min",
) -> dict[str, Any]:
    """
    Select optimal RBF bandwidth gamma and regularization lambda via Stratified K-Fold CV on X_train.

    Parameters
    ----------
    X_train : np.ndarray of shape (m_train, d)
        Training feature matrix (held-out test set is never accessed).
    y_train : np.ndarray of shape (m_train,)
        Training binary labels in {-1.0, +1.0}.
    gamma_grid : Sequence[float]
        Candidate Gaussian RBF bandwidth parameters gamma > 0.
    lambda_grid : Sequence[float]
        Candidate L2 regularization parameters lambda_reg > 0.
    model_type : str, default="pegasos"
        Exact kernel solver to evaluate ("pegasos").
    n_splits : int, default=5
        Number of stratified cross-validation folds K.
    standardize_within_folds : bool, default=False
        If True, re-standardizes each fold using only fold-training statistics.
    pegasos_iters : int, default=15000
        Number of stochastic iterations when model_type == "pegasos".
    seed : int, default=42
        Random seed for fold generation and Pegasos sampling.
    kernel_type : str, default="rbf"
        Exact kernel family ("rbf" or "laplacian"); gamma is gamma_1 for "laplacian".
    selection_rule : str, default="min"
        "min" picks the grid cell with the lowest mean CV error; "one_se" picks, among the cells whose mean
        error is within one standard error of the minimum, the most regularized one (largest lambda, then smallest gamma).

    Returns
    -------
    cv_results : dict[str, Any]
        Dictionary containing 'best_gamma', 'best_lambda', 'best_cv_error',
        'cv_error_mean_grid', and 'cv_error_std_grid'.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    gamma_list = [float(g) for g in gamma_grid]
    lambda_list = [float(l) for l in lambda_grid]

    if len(gamma_list) == 0 or len(lambda_list) == 0:
        raise ValueError("gamma_grid and lambda_grid must be non-empty.")
    if any(g <= 0.0 for g in gamma_list):
        raise ValueError("All gamma candidates must be strictly positive.")
    if any(l <= 0.0 for l in lambda_list):
        raise ValueError("All lambda candidates must be strictly positive.")
    if model_type != "pegasos":
        raise ValueError(f"model_type must be 'pegasos', got '{model_type}'.")
    if kernel_type not in ("rbf", "laplacian"):
        raise ValueError(f"kernel_type must be 'rbf' or 'laplacian', got '{kernel_type}'.")
    if selection_rule not in ("min", "one_se"):
        raise ValueError(f"selection_rule must be 'min' or 'one_se', got '{selection_rule}'.")
    gram_fn = compute_rbf_gram_matrix if kernel_type == "rbf" else compute_laplacian_gram_matrix

    folds = stratified_kfold_indices(y_train, n_splits=n_splits, seed=seed)
    n_g = len(gamma_list)
    n_l = len(lambda_list)
    cv_error_mean = np.zeros((n_g, n_l), dtype=np.float64)
    cv_error_std = np.zeros((n_g, n_l), dtype=np.float64)

    for i_g, gamma in enumerate(gamma_list):
        if not standardize_within_folds:
            K_full = gram_fn(X_train, None, gamma)
        else:
            K_full = None

        for i_l, lam in enumerate(lambda_list):
            fold_errors = np.zeros(n_splits, dtype=np.float64)
            for k_idx, (tr_idx, val_idx) in enumerate(folds):
                y_tr_fold = y_train[tr_idx]
                y_val_fold = y_train[val_idx]

                if standardize_within_folds:
                    X_tr_fold, X_val_fold, _, _ = standardize_features_zero_leakage(
                        X_train[tr_idx], X_train[val_idx]
                    )
                    K_tr_fold = gram_fn(X_tr_fold, None, gamma)
                    K_val_fold = gram_fn(X_val_fold, X_tr_fold, gamma)
                else:
                    X_tr_fold = X_train[tr_idx]
                    X_val_fold = X_train[val_idx]
                    assert K_full is not None
                    K_tr_fold = K_full[np.ix_(tr_idx, tr_idx)]
                    K_val_fold = K_full[np.ix_(val_idx, tr_idx)]

                model_svm = KernelPegasosSVM(
                    lambda_reg=lam,
                    gamma=gamma,
                    kernel_type=kernel_type,
                    n_iters=pegasos_iters,
                    seed=seed + k_idx,
                )
                model_svm.fit(X_tr_fold, y_tr_fold, K_precomputed=K_tr_fold)
                y_val_pred = model_svm.predict(X_val_fold, K_cross=K_val_fold)
                fold_errors[k_idx] = compute_classification_error(y_val_fold, y_val_pred)

            cv_error_mean[i_g, i_l] = float(np.mean(fold_errors))
            cv_error_std[i_g, i_l] = float(np.std(fold_errors))

    min_flat_idx = int(np.argmin(cv_error_mean))
    min_i_g, min_i_l = np.unravel_index(min_flat_idx, cv_error_mean.shape)
    one_se_threshold = float(
        cv_error_mean[min_i_g, min_i_l] + cv_error_std[min_i_g, min_i_l] / np.sqrt(max(n_splits - 1, 1))
    )

    if selection_rule == "min":
        best_i_g, best_i_l = min_i_g, min_i_l
    else:
        eligible = [
            (i_g, i_l)
            for i_g in range(n_g)
            for i_l in range(n_l)
            if cv_error_mean[i_g, i_l] <= one_se_threshold + 1e-12
        ]
        best_i_g, best_i_l = min(eligible, key=lambda c: (-lambda_list[c[1]], gamma_list[c[0]]))

    return {
        "best_gamma": gamma_list[best_i_g],
        "best_lambda": lambda_list[best_i_l],
        "best_cv_error": float(cv_error_mean[best_i_g, best_i_l]),
        "best_cv_std": float(cv_error_std[best_i_g, best_i_l]),
        "min_rule_gamma": gamma_list[min_i_g],
        "min_rule_lambda": lambda_list[min_i_l],
        "min_rule_cv_error": float(cv_error_mean[min_i_g, min_i_l]),
        "one_se_threshold": one_se_threshold,
        "selection_rule": selection_rule,
        "gamma_grid": gamma_list,
        "lambda_grid": lambda_list,
        "cv_error_mean_grid": cv_error_mean,
        "cv_error_std_grid": cv_error_std,
        "model_type": model_type,
        "kernel_type": kernel_type,
        "n_splits": n_splits,
    }


def prepare_core_datasets(
    m_synthetic: int = 1500,
    test_ratio: float = 0.30,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """
    Load and split both benchmark datasets (Synthetic Concentric Rings d=2 and UCI Digits d=64) with zero leakage.

    Parameters
    ----------
    m_synthetic : int, default=1500
        Total number of points generated for the Synthetic Concentric Rings dataset.
    test_ratio : float, default=0.30
        Fraction of samples held out for testing in the stratified split.
    seed : int, default=42
        Master random seed for dataset synthesis and train/test splitting.

    Returns
    -------
    datasets_dict : dict[str, dict[str, Any]]
        Dictionary keyed by 'Synthetic_d2' and 'Digits_d64' containing train/test arrays,
        metadata, split_info, and validated default hyperparameters.
    """
    X_syn, y_syn, meta_syn = generate_synthetic_concentric_rings(
        m_total=m_synthetic,
        r_0=1.436,
        sigma_noise=0.35,
        box_half_width=1.8,
        seed=seed,
    )
    X_tr_syn, X_te_syn, y_tr_syn, y_te_syn, split_syn = prepare_dataset_zero_leakage(
        X_syn,
        y_syn,
        test_ratio=test_ratio,
        standardize=False,
        seed=seed,
    )

    X_dig, y_dig, meta_dig = load_uci_digits_even_odd()
    X_tr_dig, X_te_dig, y_tr_dig, y_te_dig, split_dig = prepare_dataset_zero_leakage(
        X_dig,
        y_dig,
        test_ratio=test_ratio,
        standardize=True,
        seed=seed,
    )

    r0_syn = float(meta_syn["r_0"])
    oracle_tr_syn = float(
        np.mean(np.where(np.sum(X_tr_syn ** 2, axis=1) >= r0_syn ** 2, 1.0, -1.0) != y_tr_syn)
    )
    oracle_te_syn = float(
        np.mean(np.where(np.sum(X_te_syn ** 2, axis=1) >= r0_syn ** 2, 1.0, -1.0) != y_te_syn)
    )

    return {
        "Synthetic_d2": {
            "dataset_key": "Synthetic_d2",
            "display_name": "Synthetic Concentric Rings (d=2)",
            "X_train": X_tr_syn,
            "X_test": X_te_syn,
            "y_train": y_tr_syn,
            "y_test": y_te_syn,
            "metadata": meta_syn,
            "split_info": split_syn,
            "gamma_rbf": 1.0,
            "gamma_laplacian": 1.0,
            "lambda_svm": 1e-3,
            "bayes_risk": float(meta_syn["bayes_risk"]),
            "oracle_train_error": oracle_tr_syn,
            "oracle_test_error": oracle_te_syn,
        },
        "Digits_d64": {
            "dataset_key": "Digits_d64",
            "display_name": "UCI Handwritten Digits Even-vs-Odd (d=64)",
            "X_train": X_tr_dig,
            "X_test": X_te_dig,
            "y_train": y_tr_dig,
            "y_test": y_te_dig,
            "metadata": meta_dig,
            "split_info": split_dig,
            "gamma_rbf": 0.015,
            "gamma_laplacian": 0.02,
            "lambda_svm": 2e-4,
            "bayes_risk": None,
            "oracle_train_error": None,
            "oracle_test_error": None,
        },
    }


def run_experiment_1_convergence_vs_D(
    datasets_dict: dict[str, dict[str, Any]] | None = None,
    D_grid: Sequence[int] = (5, 10, 25, 50, 100, 250, 500, 1000, 2000),
    num_seeds: int = 10,
    pegasos_iters: int = 20000,
    base_seed: int = 1000,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Execute Core Experiment 1: Gram Matrix Approximation and Classification Error vs. D on all datasets.

    Parameters
    ----------
    datasets_dict : dict[str, dict[str, Any]] or None, default=None
        Prepared datasets dictionary. If None, calls prepare_core_datasets().
    D_grid : Sequence[int], default=(5, 10, 25, 50, 100, 250, 500, 1000, 2000)
        Sequence of Random Fourier Feature dimensions D to evaluate.
    num_seeds : int, default=10
        Number of independent Monte Carlo trials per dimension D.
    pegasos_iters : int, default=20000
        Number of stochastic subgradient iterations for Pegasos SVM solvers.
    base_seed : int, default=1000
        Base random seed offset for Monte Carlo trials.
    save_path : str, Path, or None, default=None
        Optional JSON file path to serialize scalar and mean/std results.

    Returns
    -------
    results_exp1 : dict[str, Any]
        Nested dictionary containing baseline errors, multi-seed RFF arrays,
        mean/std summaries, and empirical log-log convergence slopes for each dataset.
    """
    if datasets_dict is None:
        datasets_dict = prepare_core_datasets()

    D_list = [int(d_val) for d_val in D_grid]
    if len(D_list) == 0 or any(d_val <= 0 for d_val in D_list):
        raise ValueError("D_grid must be a non-empty sequence of positive integers.")
    if num_seeds <= 0:
        raise ValueError(f"num_seeds must be strictly positive, got {num_seeds}.")

    n_D = len(D_list)
    results_exp1: dict[str, Any] = {
        "experiment_id": "Exp1_Convergence_vs_D",
        "D_grid": D_list,
        "num_seeds": int(num_seeds),
        "pegasos_iters": int(pegasos_iters),
        "datasets": {},
    }

    for ds_key, ds_bundle in datasets_dict.items():
        X_train = np.asarray(ds_bundle["X_train"], dtype=np.float64)
        X_test = np.asarray(ds_bundle["X_test"], dtype=np.float64)
        y_train = np.asarray(ds_bundle["y_train"], dtype=np.float64)
        y_test = np.asarray(ds_bundle["y_test"], dtype=np.float64)
        gamma = float(ds_bundle["gamma_rbf"])
        lam_svm = float(ds_bundle["lambda_svm"])
        bayes_risk = ds_bundle.get("bayes_risk", None)
        oracle_train_err = ds_bundle.get("oracle_train_error", None)
        oracle_test_err = ds_bundle.get("oracle_test_error", None)

        lin_svm = PrimalPegasosSVM(
            lambda_reg=lam_svm,
            n_iters=pegasos_iters,
            fit_intercept=True,
            seed=base_seed,
        ).fit(X_train, y_train)
        lin_svm_train_err = compute_classification_error(y_train, lin_svm.predict(X_train))
        lin_svm_test_err = compute_classification_error(y_test, lin_svm.predict(X_test))

        K_train = compute_rbf_gram_matrix(X_train, None, gamma=gamma)
        K_test_train = compute_rbf_gram_matrix(X_test, X_train, gamma=gamma)

        ker_svm = KernelPegasosSVM(
            lambda_reg=lam_svm,
            gamma=gamma,
            kernel_type="rbf",
            n_iters=pegasos_iters,
            seed=base_seed,
        ).fit(X_train, y_train, K_precomputed=K_train)
        ker_svm_train_err = compute_classification_error(
            y_train, ker_svm.predict(X_train, K_cross=K_train)
        )
        ker_svm_test_err = compute_classification_error(
            y_test, ker_svm.predict(X_test, K_cross=K_test_train)
        )

        frob_rel_mat = np.zeros((n_D, num_seeds), dtype=np.float64)
        max_abs_mat = np.zeros((n_D, num_seeds), dtype=np.float64)
        mean_abs_mat = np.zeros((n_D, num_seeds), dtype=np.float64)
        rff_svm_train_mat = np.zeros((n_D, num_seeds), dtype=np.float64)
        rff_svm_test_mat = np.zeros((n_D, num_seeds), dtype=np.float64)

        for i_d, D in enumerate(D_list):
            for s in range(num_seeds):
                seed_rff = base_seed + i_d * 100 + s
                seed_svm = base_seed + 10000 + i_d * 100 + s

                rff = RFFCosinePhase(D=D, gamma=gamma, kernel_type="rbf", seed=seed_rff)
                Z_train = rff.fit_transform(X_train)
                Z_test = rff.transform(X_test)

                K_approx = Z_train @ Z_train.T
                gram_errs = kernel_matrix_approximation_errors(K_train, K_approx)
                frob_rel_mat[i_d, s] = gram_errs["frobenius_relative"]
                max_abs_mat[i_d, s] = gram_errs["max_abs_error"]
                mean_abs_mat[i_d, s] = gram_errs["mean_abs_error"]

                rff_svm = PrimalPegasosSVM(
                    lambda_reg=lam_svm,
                    n_iters=pegasos_iters,
                    fit_intercept=False,
                    seed=seed_svm,
                ).fit(Z_train, y_train)
                rff_svm_train_mat[i_d, s] = compute_classification_error(
                    y_train, rff_svm.predict(Z_train)
                )
                rff_svm_test_mat[i_d, s] = compute_classification_error(
                    y_test, rff_svm.predict(Z_test)
                )

        frob_rel_mean = np.mean(frob_rel_mat, axis=1)
        frob_rel_std = np.std(frob_rel_mat, axis=1)
        max_abs_mean = np.mean(max_abs_mat, axis=1)
        max_abs_std = np.std(max_abs_mat, axis=1)
        mean_abs_mean = np.mean(mean_abs_mat, axis=1)
        mean_abs_std = np.std(mean_abs_mat, axis=1)

        frob_slope = _fit_loglog_slope(np.asarray(D_list), frob_rel_mean) if n_D >= 2 else 0.0
        sup_slope = _fit_loglog_slope(np.asarray(D_list), max_abs_mean) if n_D >= 2 else 0.0

        results_exp1["datasets"][ds_key] = {
            "dataset_key": ds_key,
            "display_name": ds_bundle.get("display_name", ds_key),
            "m_train": int(X_train.shape[0]),
            "m_test": int(X_test.shape[0]),
            "d": int(X_train.shape[1]),
            "gamma_rbf": gamma,
            "lambda_svm": lam_svm,
            "bayes_risk": float(bayes_risk) if bayes_risk is not None else None,
            "oracle_train_error": float(oracle_train_err) if oracle_train_err is not None else None,
            "oracle_test_error": float(oracle_test_err) if oracle_test_err is not None else None,
            "D_grid": D_list,
            "baselines": {
                "linear_svm_train_error": lin_svm_train_err,
                "linear_svm_test_error": lin_svm_test_err,
                "exact_kernel_svm_train_error": ker_svm_train_err,
                "exact_kernel_svm_test_error": ker_svm_test_err,
            },
            "gram_frobenius_rel_mean": frob_rel_mean,
            "gram_frobenius_rel_std": frob_rel_std,
            "gram_max_abs_mean": max_abs_mean,
            "gram_max_abs_std": max_abs_std,
            "gram_mean_abs_mean": mean_abs_mean,
            "gram_mean_abs_std": mean_abs_std,
            "rff_svm_train_mean": np.mean(rff_svm_train_mat, axis=1),
            "rff_svm_train_std": np.std(rff_svm_train_mat, axis=1),
            "rff_svm_test_mean": np.mean(rff_svm_test_mat, axis=1),
            "rff_svm_test_std": np.std(rff_svm_test_mat, axis=1),
            "empirical_loglog_slope_frobenius": frob_slope,
            "empirical_loglog_slope_supremum": sup_slope,
            "raw_trials": {
                "frob_rel": frob_rel_mat,
                "max_abs": max_abs_mat,
                "rff_svm_train": rff_svm_train_mat,
                "rff_svm_test": rff_svm_test_mat,
            },
        }

    if save_path is not None:
        _save_json_compatible(results_exp1, Path(save_path))

    return results_exp1


def run_experiment_2_decision_boundaries_2d(
    dataset_syn: dict[str, Any] | None = None,
    D_list: Sequence[int] = (5, 15, 50, 500),
    grid_resolution: int = 200,
    grid_bound: float = 2.0,
    model_type: str = "pegasos",
    pegasos_iters: int = 25000,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Execute Core Experiment 2: 2D Decision Boundary and Margin Gutter Progression on Synthetic Concentric Rings.

    Parameters
    ----------
    dataset_syn : dict[str, Any] or None, default=None
        Prepared Synthetic_d2 dataset bundle. If None, loads from prepare_core_datasets().
    D_list : Sequence[int], default=(5, 15, 50, 500)
        Sequence of RFF dimensions D for the intermediate progression panels.
    grid_resolution : int, default=200
        Number of grid points per coordinate axis for the 2D contour meshgrid.
    grid_bound : float, default=2.0
        Half-width of the 2D evaluation box [-grid_bound, +grid_bound]^2.
    model_type : str, default="pegasos"
        Classifier family used to generate the decision surfaces ("pegasos").
    pegasos_iters : int, default=25000
        Number of stochastic iterations when model_type == "pegasos".
    seed : int, default=42
        Random seed for RFF frequency sampling and Pegasos optimization.
    save_path : str, Path, or None, default=None
        Optional JSON file path to serialize panel metrics.

    Returns
    -------
    results_exp2 : dict[str, Any]
        Dictionary containing the 2D meshgrid (xx, yy), data splits, true circle radius r_0,
        and ordered panel dictionaries for Linear, RFF(D), and Exact RBF Kernel models.
    """
    if dataset_syn is None:
        dataset_syn = prepare_core_datasets(seed=seed)["Synthetic_d2"]

    if grid_resolution < 10:
        raise ValueError(f"grid_resolution must be at least 10, got {grid_resolution}.")
    if grid_bound <= 0.0:
        raise ValueError(f"grid_bound must be strictly positive, got {grid_bound}.")
    if model_type != "pegasos":
        raise ValueError(f"model_type must be 'pegasos', got '{model_type}'.")

    D_seq = [int(d_val) for d_val in D_list]
    if len(D_seq) == 0 or any(d_val <= 0 for d_val in D_seq):
        raise ValueError("D_list must be a non-empty sequence of positive integers.")

    X_train = np.asarray(dataset_syn["X_train"], dtype=np.float64)
    X_test = np.asarray(dataset_syn["X_test"], dtype=np.float64)
    y_train = np.asarray(dataset_syn["y_train"], dtype=np.float64)
    y_test = np.asarray(dataset_syn["y_test"], dtype=np.float64)
    if X_train.shape[1] != 2:
        raise ValueError(f"Experiment 2 requires 2D input features, got d={X_train.shape[1]}.")

    gamma = float(dataset_syn["gamma_rbf"])
    lam = float(dataset_syn["lambda_svm"])
    meta = dataset_syn.get("metadata", {})
    r_0 = float(meta.get("r_0", 1.436))
    bayes_risk = float(dataset_syn.get("bayes_risk", meta.get("bayes_risk", 0.066)))
    oracle_train_err = float(
        np.mean(np.where(np.sum(X_train ** 2, axis=1) >= r_0 ** 2, 1.0, -1.0) != y_train)
    )
    oracle_test_err = float(
        np.mean(np.where(np.sum(X_test ** 2, axis=1) >= r_0 ** 2, 1.0, -1.0) != y_test)
    )

    axis_coords = np.linspace(-grid_bound, grid_bound, grid_resolution, dtype=np.float64)
    xx, yy = np.meshgrid(axis_coords, axis_coords)
    X_grid = np.column_stack([xx.ravel(), yy.ravel()])

    K_train = compute_rbf_gram_matrix(X_train, None, gamma=gamma)
    K_test_train = compute_rbf_gram_matrix(X_test, X_train, gamma=gamma)
    K_grid_train = compute_rbf_gram_matrix(X_grid, X_train, gamma=gamma)

    exact_model = KernelPegasosSVM(
        lambda_reg=lam,
        gamma=gamma,
        kernel_type="rbf",
        n_iters=pegasos_iters,
        seed=seed,
    ).fit(X_train, y_train, K_precomputed=K_train)
    exact_grid_scores = exact_model.decision_function(X_grid, K_cross=K_grid_train)
    exact_train_err = compute_classification_error(
        y_train, exact_model.predict(X_train, K_cross=K_train)
    )
    exact_test_err = compute_classification_error(
        y_test, exact_model.predict(X_test, K_cross=K_test_train)
    )

    lin_model = PrimalPegasosSVM(
        lambda_reg=lam,
        n_iters=pegasos_iters,
        fit_intercept=True,
        seed=seed,
    ).fit(X_train, y_train)
    lin_grid_scores = lin_model.decision_function(X_grid)
    lin_train_err = compute_classification_error(y_train, lin_model.predict(X_train))
    lin_test_err = compute_classification_error(y_test, lin_model.predict(X_test))

    panels: list[dict[str, Any]] = [
        {
            "panel_key": "Linear_d2",
            "title": "Linear Baseline ($d=2$)",
            "D": None,
            "score_grid": lin_grid_scores.reshape(grid_resolution, grid_resolution),
            "train_error": lin_train_err,
            "test_error": lin_test_err,
            "gram_frobenius_rel": None,
            "grid_score_rmse_vs_exact": float(
                np.sqrt(np.mean((lin_grid_scores - exact_grid_scores) ** 2))
            ),
        }
    ]

    for D in D_seq:
        rff = RFFCosinePhase(D=D, gamma=gamma, kernel_type="rbf", seed=seed + D)
        Z_train = rff.fit_transform(X_train)
        Z_test = rff.transform(X_test)
        Z_grid = rff.transform(X_grid)

        K_approx = Z_train @ Z_train.T
        gram_frob_rel = kernel_matrix_approximation_errors(K_train, K_approx)["frobenius_relative"]

        rff_model = PrimalPegasosSVM(
            lambda_reg=lam,
            n_iters=pegasos_iters,
            fit_intercept=False,
            seed=seed,
        ).fit(Z_train, y_train)
        rff_grid_scores = rff_model.decision_function(Z_grid)
        rff_train_err = compute_classification_error(y_train, rff_model.predict(Z_train))
        rff_test_err = compute_classification_error(y_test, rff_model.predict(Z_test))

        panels.append(
            {
                "panel_key": f"RFF_D{D}",
                "title": f"RFF + Linear ($D={D}$)",
                "D": int(D),
                "score_grid": rff_grid_scores.reshape(grid_resolution, grid_resolution),
                "train_error": rff_train_err,
                "test_error": rff_test_err,
                "gram_frobenius_rel": gram_frob_rel,
                "grid_score_rmse_vs_exact": float(
                    np.sqrt(np.mean((rff_grid_scores - exact_grid_scores) ** 2))
                ),
            }
        )

    panels.append(
        {
            "panel_key": "Exact_RBF_Kernel",
            "title": r"Exact RBF Kernel ($\mathcal{H}_K, D=\infty$)",
            "D": None,
            "score_grid": exact_grid_scores.reshape(grid_resolution, grid_resolution),
            "train_error": exact_train_err,
            "test_error": exact_test_err,
            "gram_frobenius_rel": 0.0,
            "grid_score_rmse_vs_exact": 0.0,
        }
    )

    results_exp2: dict[str, Any] = {
        "experiment_id": "Exp2_Decision_Boundaries_2D",
        "model_type": model_type,
        "gamma_rbf": gamma,
        "lambda_reg": lam,
        "r_0": r_0,
        "bayes_risk": bayes_risk,
        "oracle_train_error": oracle_train_err,
        "oracle_test_error": oracle_test_err,
        "grid_resolution": int(grid_resolution),
        "grid_bound": float(grid_bound),
        "D_list": D_seq,
        "xx": xx,
        "yy": yy,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "panels": panels,
    }

    if save_path is not None:
        summary_only = {
            k: v
            for k, v in results_exp2.items()
            if k not in ("xx", "yy", "X_train", "y_train", "X_test", "y_test", "panels")
        }
        summary_only["panels"] = [
            {pk: pv for pk, pv in p.items() if pk != "score_grid"} for p in panels
        ]
        _save_json_compatible(summary_only, Path(save_path))

    return results_exp2


DEFAULT_CV_GRIDS: dict[str, dict[str, list[float]]] = {
    "Synthetic_d2": {
        "gamma_rbf": [0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
        "lambda_svm": [2e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "gamma_laplacian": [0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
    },
    "Digits_d64": {
        "gamma_rbf": [0.001, 0.002, 0.005, 0.01, 0.015],
        "lambda_svm": [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 5e-3],
        "gamma_laplacian": [0.005, 0.01, 0.015, 0.02],
    },
}

DEFAULT_EXP3_GAMMA_GRIDS: dict[str, list[float]] = {
    "Synthetic_d2": [1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0, 10.0, 50.0, 200.0, 1000.0],
    "Digits_d64": [1e-4, 5e-4, 2e-3, 8e-3, 1.5e-2, 3e-2, 1e-1, 3e-1, 1.0],
}


def _is_on_grid_boundary(value: float, grid: Sequence[float]) -> bool:
    """
    Check whether a selected hyperparameter equals the smallest or largest grid candidate.

    Parameters
    ----------
    value : float
        Selected hyperparameter value.
    grid : Sequence[float]
        Candidate grid the value was selected from.

    Returns
    -------
    on_boundary : bool
        True if value is the minimum or maximum of grid (and grid has more than one element).
    """
    grid_arr = np.asarray(grid, dtype=np.float64)
    if grid_arr.size < 2:
        return False
    return bool(np.isclose(value, grid_arr.min()) or np.isclose(value, grid_arr.max()))


def select_hyperparameters_by_cv(
    datasets_dict: dict[str, dict[str, Any]],
    cv_grids: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    n_splits: int = 5,
    pegasos_iters: int = 20000,
    seed: int = 42,
    selection_rule: str = "min",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """
    Select (gamma_rbf, lambda_svm) jointly and gamma_laplacian with Kernel Pegasos SVM via CV on training splits only.

    Parameters
    ----------
    datasets_dict : dict[str, dict[str, Any]]
        Output of prepare_core_datasets(); test splits are never accessed.
    cv_grids : dict[str, dict[str, Sequence[float]]] or None, default=None
        Candidate grids per dataset; defaults to DEFAULT_CV_GRIDS.
    n_splits : int, default=5
        Number of stratified cross-validation folds.
    pegasos_iters : int, default=20000
        Pegasos iterations used when jointly selecting (gamma_rbf, lambda_svm).
    seed : int, default=42
        Random seed for fold generation and Pegasos sampling.
    selection_rule : str, default="min"
        Grid-cell selection rule forwarded to cross_validate_exact_kernel ("min" or "one_se").

    Returns
    -------
    updated_datasets : dict[str, dict[str, Any]]
        Copy of datasets_dict whose bundles carry the CV-selected hyperparameters.
    report : dict[str, Any]
        Per-dataset selected values, CV errors, and grid-boundary flags.
    """
    grids_all = DEFAULT_CV_GRIDS if cv_grids is None else cv_grids
    updated: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {"n_splits": int(n_splits), "selection_rule": selection_rule, "datasets": {}}

    for ds_key, bundle in datasets_dict.items():
        if ds_key not in grids_all:
            raise KeyError(f"No CV grid specified for dataset '{ds_key}'.")
        grids = grids_all[ds_key]
        X_tr = bundle["X_train"]
        y_tr = bundle["y_train"]
        std_in_folds = bool(bundle["split_info"]["standardize"])

        cv_svm = cross_validate_exact_kernel(
            X_tr, y_tr, grids["gamma_rbf"], grids["lambda_svm"], model_type="pegasos",
            n_splits=n_splits, standardize_within_folds=std_in_folds,
            pegasos_iters=pegasos_iters, seed=seed, kernel_type="rbf", selection_rule=selection_rule,
        )
        cv_lap = cross_validate_exact_kernel(
            X_tr, y_tr, grids["gamma_laplacian"], [cv_svm["best_lambda"]], model_type="pegasos",
            n_splits=n_splits, standardize_within_folds=std_in_folds, seed=seed,
            kernel_type="laplacian", pegasos_iters=pegasos_iters, selection_rule=selection_rule,
        )

        new_bundle = dict(bundle)
        new_bundle["gamma_rbf"] = float(cv_svm["best_gamma"])
        new_bundle["lambda_svm"] = float(cv_svm["best_lambda"])
        new_bundle["gamma_laplacian"] = float(cv_lap["best_gamma"])
        updated[ds_key] = new_bundle

        report["datasets"][ds_key] = {
            "gamma_rbf": new_bundle["gamma_rbf"],
            "lambda_svm": new_bundle["lambda_svm"],
            "gamma_laplacian": new_bundle["gamma_laplacian"],
            "cv_error_svm_rbf": cv_svm["best_cv_error"],
            "cv_error_svm_laplacian": cv_lap["best_cv_error"],
            "min_rule_choice": {
                "gamma_rbf": cv_svm["min_rule_gamma"],
                "lambda_svm": cv_svm["min_rule_lambda"],
                "gamma_laplacian": cv_lap["min_rule_gamma"],
            },
            "grids": {k: [float(v) for v in vals] for k, vals in grids.items()},
            "on_grid_boundary": {
                "gamma_rbf": _is_on_grid_boundary(new_bundle["gamma_rbf"], grids["gamma_rbf"]),
                "lambda_svm": _is_on_grid_boundary(new_bundle["lambda_svm"], grids["lambda_svm"]),
                "gamma_laplacian": _is_on_grid_boundary(new_bundle["gamma_laplacian"], grids["gamma_laplacian"]),
            },
            "cv_error_mean_grid_svm_rbf": cv_svm["cv_error_mean_grid"],
            "cv_error_mean_grid_svm_laplacian": cv_lap["cv_error_mean_grid"],
        }

    return updated, report


def run_experiment_3_bandwidth_effect(
    datasets_dict: dict[str, dict[str, Any]] | None = None,
    gamma_grids: dict[str, Sequence[float]] | None = None,
    D_subset: Sequence[int] = (25, 100, 500),
    num_seeds: int = 10,
    pegasos_iters: int = 15000,
    base_seed: int = 3000,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Execute Extension 1: effect of the RBF bandwidth gamma on exact and RFF Pegasos SVM classifiers (both datasets).

    Parameters
    ----------
    datasets_dict : dict[str, dict[str, Any]] or None, default=None
        Prepared datasets; if None, calls prepare_core_datasets().
    gamma_grids : dict[str, Sequence[float]] or None, default=None
        Bandwidth grids per dataset; defaults to DEFAULT_EXP3_GAMMA_GRIDS.
    D_subset : Sequence[int], default=(25, 100, 500)
        RFF dimensions compared against the exact kernel machine.
    num_seeds : int, default=10
        Number of independent RFF draws per (gamma, D).
    pegasos_iters : int, default=15000
        Stochastic iterations for Pegasos optimization.
    base_seed : int, default=3000
        Base seed offset.
    save_path : str, Path, or None, default=None
        Optional JSON output path.

    Returns
    -------
    results_exp3 : dict[str, Any]
        Per-dataset arrays of exact train/test error, and RFF Gram/train/test error
        (mean and std over seeds) indexed by (gamma, D).
    """
    if datasets_dict is None:
        datasets_dict = prepare_core_datasets()
    grids_all = DEFAULT_EXP3_GAMMA_GRIDS if gamma_grids is None else gamma_grids
    D_list = [int(v) for v in D_subset]
    if len(D_list) == 0 or any(v <= 0 for v in D_list):
        raise ValueError("D_subset must be a non-empty sequence of positive integers.")
    if num_seeds <= 0:
        raise ValueError(f"num_seeds must be strictly positive, got {num_seeds}.")

    results: dict[str, Any] = {
        "experiment_id": "Exp3_Bandwidth_Effect",
        "D_subset": D_list,
        "num_seeds": int(num_seeds),
        "datasets": {},
    }

    for ds_key, bundle in datasets_dict.items():
        gamma_list = [float(g) for g in grids_all[ds_key]]
        if any(g <= 0.0 for g in gamma_list):
            raise ValueError("All gamma candidates must be strictly positive.")
        X_tr = np.asarray(bundle["X_train"], dtype=np.float64)
        X_te = np.asarray(bundle["X_test"], dtype=np.float64)
        y_tr = np.asarray(bundle["y_train"], dtype=np.float64)
        y_te = np.asarray(bundle["y_test"], dtype=np.float64)
        lam = float(bundle["lambda_svm"])
        d_dim = int(X_tr.shape[1])
        n_g, n_D = len(gamma_list), len(D_list)

        exact_tr = np.zeros(n_g)
        exact_te = np.zeros(n_g)
        frob = np.zeros((n_g, n_D, num_seeds))
        supn = np.zeros((n_g, n_D, num_seeds))
        rff_tr = np.zeros((n_g, n_D, num_seeds))
        rff_te = np.zeros((n_g, n_D, num_seeds))

        for i_g, gamma in enumerate(gamma_list):
            K_tr = compute_rbf_gram_matrix(X_tr, None, gamma=gamma)
            K_te = compute_rbf_gram_matrix(X_te, X_tr, gamma=gamma)
            kr = KernelPegasosSVM(
                lambda_reg=lam, gamma=gamma, kernel_type="rbf", n_iters=pegasos_iters, seed=base_seed + i_g
            ).fit(X_tr, y_tr, K_precomputed=K_tr)
            exact_tr[i_g] = compute_classification_error(y_tr, kr.predict(X_tr, K_cross=K_tr))
            exact_te[i_g] = compute_classification_error(y_te, kr.predict(X_te, K_cross=K_te))

            for i_d, D in enumerate(D_list):
                for s in range(num_seeds):
                    rff = RFFCosinePhase(
                        D=D, gamma=gamma, kernel_type="rbf",
                        seed=base_seed + i_g * 1000 + i_d * 100 + s,
                    )
                    Z_tr = rff.fit_transform(X_tr)
                    Z_te = rff.transform(X_te)
                    errs = kernel_matrix_approximation_errors(K_tr, Z_tr @ Z_tr.T)
                    frob[i_g, i_d, s] = errs["frobenius_relative"]
                    supn[i_g, i_d, s] = errs["max_abs_error"]
                    clf = PrimalPegasosSVM(
                        lambda_reg=lam,
                        fit_intercept=False,
                        n_iters=pegasos_iters,
                        seed=base_seed + 10000 + i_g * 1000 + i_d * 100 + s,
                    ).fit(Z_tr, y_tr)
                    rff_tr[i_g, i_d, s] = compute_classification_error(y_tr, clf.predict(Z_tr))
                    rff_te[i_g, i_d, s] = compute_classification_error(y_te, clf.predict(Z_te))

        results["datasets"][ds_key] = {
            "dataset_key": ds_key,
            "display_name": bundle.get("display_name", ds_key),
            "d": d_dim,
            "m_train": int(X_tr.shape[0]),
            "lambda_svm": lam,
            "gamma_default": float(bundle["gamma_rbf"]),
            "gamma_grid": gamma_list,
            "sigma_p_squared": [2.0 * d_dim * g for g in gamma_list],
            "D_subset": D_list,
            "exact_train_error": exact_tr,
            "exact_test_error": exact_te,
            "gram_frobenius_rel_mean": frob.mean(axis=2),
            "gram_frobenius_rel_std": frob.std(axis=2),
            "gram_max_abs_mean": supn.mean(axis=2),
            "gram_max_abs_std": supn.std(axis=2),
            "rff_train_mean": rff_tr.mean(axis=2),
            "rff_train_std": rff_tr.std(axis=2),
            "rff_test_mean": rff_te.mean(axis=2),
            "rff_test_std": rff_te.std(axis=2),
        }

    if save_path is not None:
        _save_json_compatible(results, Path(save_path))
    return results


def run_experiment_4_feature_map_comparison(
    datasets_dict: dict[str, dict[str, Any]] | None = None,
    D_grid: Sequence[int] = (10, 20, 50, 100, 250, 500, 1000),
    P_grid: Sequence[int] = (1, 2, 5, 10, 20, 50, 100, 200),
    num_seeds: int = 10,
    pegasos_iters: int = 15000,
    base_seed: int = 4000,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Execute Extension 2: compare feature-map constructions for the RBF and Laplacian kernels (both datasets).

    Parameters
    ----------
    datasets_dict : dict[str, dict[str, Any]] or None, default=None
        Prepared datasets; if None, calls prepare_core_datasets().
    D_grid : Sequence[int], default=(10, 20, 50, 100, 250, 500, 1000)
        Even feature budgets for the RBF maps (phase cosine, cos-sin pair, ORF).
    P_grid : Sequence[int], default=(1, 2, 5, 10, 20, 50, 100, 200)
        Budgets for the Laplacian maps: P grids for Random Binning, D=P Cauchy features for RFF.
    num_seeds : int, default=10
        Number of independent random draws per budget.
    pegasos_iters : int, default=15000
        Stochastic iterations for Pegasos optimization.
    base_seed : int, default=4000
        Base seed offset.
    save_path : str, Path, or None, default=None
        Optional JSON output path.

    Returns
    -------
    results_exp4 : dict[str, Any]
        Per-dataset Gram error and Pegasos test error (mean/std over seeds) of every construction,
        the exact-kernel Pegasos reference errors, and the mean occupied-bin count of Random Binning.
    """
    if datasets_dict is None:
        datasets_dict = prepare_core_datasets()
    D_list = [int(v) for v in D_grid]
    P_list = [int(v) for v in P_grid]
    if len(D_list) == 0 or any(v <= 0 or v % 2 != 0 for v in D_list):
        raise ValueError("D_grid must be a non-empty sequence of positive even integers.")
    if len(P_list) == 0 or any(v <= 0 for v in P_list):
        raise ValueError("P_grid must be a non-empty sequence of positive integers.")
    if num_seeds <= 0:
        raise ValueError(f"num_seeds must be strictly positive, got {num_seeds}.")

    rbf_makers = {
        "phase_cosine": lambda D, g, sd: RFFCosinePhase(D=D, gamma=g, kernel_type="rbf", seed=sd),
        "cos_sin_pair": lambda D, g, sd: RFFCosSinPair(D=D, gamma=g, seed=sd),
        "orthogonal": lambda D, g, sd: OrthogonalRandomFeatures(D=D, gamma=g, seed=sd),
    }
    lap_makers = {
        "random_binning": lambda P, g, sd: RandomBinningFeatures(P_grids=P, gamma_1=g, seed=sd),
        "cauchy_rff": lambda P, g, sd: RFFCosinePhase(D=P, gamma=g, kernel_type="laplacian", seed=sd),
    }

    results: dict[str, Any] = {
        "experiment_id": "Exp4_Feature_Map_Comparison",
        "D_grid": D_list,
        "P_grid": P_list,
        "num_seeds": int(num_seeds),
        "datasets": {},
    }

    def _sweep(makers, budgets, K_exact, X_tr, X_te, y_tr, y_te, gamma, lam, seed_off):
        out: dict[str, Any] = {}
        for name, make in makers.items():
            frob = np.zeros((len(budgets), num_seeds))
            te = np.zeros((len(budgets), num_seeds))
            tr = np.zeros((len(budgets), num_seeds))
            dim = np.zeros((len(budgets), num_seeds))
            for i_b, B in enumerate(budgets):
                for s in range(num_seeds):
                    fm = make(B, gamma, seed_off + i_b * 100 + s)
                    Z_tr = fm.fit_transform(X_tr)
                    Z_te = fm.transform(X_te)
                    frob[i_b, s] = kernel_matrix_approximation_errors(K_exact, Z_tr @ Z_tr.T)["frobenius_relative"]
                    clf = PrimalPegasosSVM(
                        lambda_reg=lam, fit_intercept=False, n_iters=pegasos_iters, seed=seed_off + i_b * 100 + s
                    ).fit(Z_tr, y_tr)
                    tr[i_b, s] = compute_classification_error(y_tr, clf.predict(Z_tr))
                    te[i_b, s] = compute_classification_error(y_te, clf.predict(Z_te))
                    dim[i_b, s] = Z_tr.shape[1]
            out[name] = {
                "gram_frobenius_rel_mean": frob.mean(axis=1),
                "gram_frobenius_rel_std": frob.std(axis=1),
                "train_error_mean": tr.mean(axis=1),
                "test_error_mean": te.mean(axis=1),
                "test_error_std": te.std(axis=1),
                "feature_dimension_mean": dim.mean(axis=1),
            }
        return out

    for ds_key, bundle in datasets_dict.items():
        X_tr = np.asarray(bundle["X_train"], dtype=np.float64)
        X_te = np.asarray(bundle["X_test"], dtype=np.float64)
        y_tr = np.asarray(bundle["y_train"], dtype=np.float64)
        y_te = np.asarray(bundle["y_test"], dtype=np.float64)
        g_rbf = float(bundle["gamma_rbf"])
        g_lap = float(bundle["gamma_laplacian"])
        lam = float(bundle["lambda_svm"])

        K_rbf = compute_rbf_gram_matrix(X_tr, None, gamma=g_rbf)
        K_rbf_te = compute_rbf_gram_matrix(X_te, X_tr, gamma=g_rbf)
        K_lap = compute_laplacian_gram_matrix(X_tr, None, gamma_1=g_lap)
        K_lap_te = compute_laplacian_gram_matrix(X_te, X_tr, gamma_1=g_lap)

        ex_rbf = KernelPegasosSVM(
            lambda_reg=lam, gamma=g_rbf, kernel_type="rbf", n_iters=pegasos_iters, seed=base_seed
        ).fit(X_tr, y_tr, K_precomputed=K_rbf)
        ex_lap = KernelPegasosSVM(
            lambda_reg=lam, gamma=g_lap, kernel_type="laplacian", n_iters=pegasos_iters, seed=base_seed
        ).fit(X_tr, y_tr, K_precomputed=K_lap)

        results["datasets"][ds_key] = {
            "dataset_key": ds_key,
            "display_name": bundle.get("display_name", ds_key),
            "d": int(X_tr.shape[1]),
            "m_train": int(X_tr.shape[0]),
            "gamma_rbf": g_rbf,
            "gamma_laplacian": g_lap,
            "lambda_svm": lam,
            "exact_rbf_test_error": compute_classification_error(y_te, ex_rbf.predict(X_te, K_cross=K_rbf_te)),
            "exact_laplacian_test_error": compute_classification_error(y_te, ex_lap.predict(X_te, K_cross=K_lap_te)),
            "D_grid": D_list,
            "P_grid": P_list,
            "rbf_maps": _sweep(rbf_makers, D_list, K_rbf, X_tr, X_te, y_tr, y_te, g_rbf, lam, base_seed),
            "laplacian_maps": _sweep(lap_makers, P_list, K_lap, X_tr, X_te, y_tr, y_te, g_lap, lam, base_seed + 50000),
        }

    if save_path is not None:
        _save_json_compatible(results, Path(save_path))
    return results


def _median_timed(fn, repeats: int) -> tuple[Any, float]:
    """
    Run a zero-argument callable several times and return its last result and the median wall-clock seconds.

    Parameters
    ----------
    fn : Callable[[], Any]
        Callable to time with time.perf_counter().
    repeats : int
        Number of timed repetitions (at least 1).

    Returns
    -------
    result : Any
        Return value of the final repetition.
    median_seconds : float
        Median elapsed wall-clock time in seconds.
    """
    times: list[float] = []
    result: Any = None
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, float(np.median(times))


def _time_exact_and_rff(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    gamma: float,
    lam_svm: float,
    D_list: Sequence[int],
    pegasos_iters: int,
    num_seeds: int,
    base_seed: int,
) -> dict[str, Any]:
    """
    Time training (feature map + solver) and inference (feature map + prediction) of exact and RFF Pegasos SVM models.

    Parameters
    ----------
    X_tr, y_tr : np.ndarray
        Training split.
    X_te, y_te : np.ndarray
        Test split.
    gamma : float
        RBF bandwidth.
    lam_svm : float
        Regularization of the Pegasos solver.
    D_list : Sequence[int]
        RFF dimensions to time.
    pegasos_iters : int
        Pegasos iterations (identical for exact and RFF).
    num_seeds : int
        Repetitions per configuration; the median time is reported.
    base_seed : int
        Base seed offset.

    Returns
    -------
    timings : dict[str, Any]
        Median times in seconds and mean test errors for exact and RFF models.
    """
    exact: dict[str, Any] = {}
    model_fn = lambda: KernelPegasosSVM(
        lambda_reg=lam_svm, gamma=gamma, kernel_type="rbf", n_iters=pegasos_iters, seed=base_seed
    )
    model, t_train = _median_timed(lambda: model_fn().fit(X_tr, y_tr), num_seeds)
    pred, t_inf = _median_timed(lambda: model.predict(X_te), num_seeds)
    exact["pegasos"] = {
        "train_time_s": t_train,
        "inference_time_s": t_inf,
        "test_error": compute_classification_error(y_te, pred),
    }

    rff: dict[str, Any] = {"pegasos": {"train_time_s": [], "inference_time_s": [], "test_error": []}}
    for i_d, D in enumerate(D_list):
        t_tr_runs, t_in_runs, errs = [], [], []
        for s in range(num_seeds):
            seed = base_seed + i_d * 100 + s
            t0 = time.perf_counter()
            fm = RFFCosinePhase(D=int(D), gamma=gamma, kernel_type="rbf", seed=seed)
            Z_tr = fm.fit_transform(X_tr)
            clf = PrimalPegasosSVM(
                lambda_reg=lam_svm, n_iters=pegasos_iters, fit_intercept=False, seed=seed
            ).fit(Z_tr, y_tr)
            t1 = time.perf_counter()
            pred = clf.predict(fm.transform(X_te))
            t2 = time.perf_counter()
            t_tr_runs.append(t1 - t0)
            t_in_runs.append(t2 - t1)
            errs.append(compute_classification_error(y_te, pred))
        rff["pegasos"]["train_time_s"].append(float(np.median(t_tr_runs)))
        rff["pegasos"]["inference_time_s"].append(float(np.median(t_in_runs)))
        rff["pegasos"]["test_error"].append(float(np.mean(errs)))

    for key in rff["pegasos"]:
        rff["pegasos"][key] = np.asarray(rff["pegasos"][key], dtype=np.float64)
    return {"exact": exact, "rff": rff}


def run_experiment_5_runtime_vs_accuracy(
    datasets_dict: dict[str, dict[str, Any]] | None = None,
    D_grid: Sequence[int] = (5, 10, 25, 50, 100, 250, 500, 1000, 2000),
    m_scaling_grid: Sequence[int] = (250, 500, 1000, 2000, 4000, 6000),
    D_fixed: int = 300,
    n_test_scaling: int = 1000,
    num_seeds: int = 3,
    pegasos_iters: int = 10000,
    base_seed: int = 5000,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Execute Extension 3: wall-clock runtime versus accuracy of exact and RFF Pegasos SVM models.

    Part A varies D at the fixed training size of each dataset; Part B varies the training size m of the
    synthetic generator at fixed D.

    Parameters
    ----------
    datasets_dict : dict[str, dict[str, Any]] or None, default=None
        Prepared datasets; if None, calls prepare_core_datasets().
    D_grid : Sequence[int], default=(5, 10, 25, 50, 100, 250, 500, 1000, 2000)
        RFF dimensions for Part A.
    m_scaling_grid : Sequence[int], default=(250, 500, 1000, 2000, 4000, 6000)
        Training sizes for Part B.
    D_fixed : int, default=300
        RFF dimension for Part B.
    n_test_scaling : int, default=1000
        Test-set size for Part B.
    num_seeds : int, default=3
        Timing repetitions per configuration (median reported).
    pegasos_iters : int, default=10000
        Pegasos iterations for all Pegasos models.
    base_seed : int, default=5000
        Base seed offset.
    save_path : str, Path, or None, default=None
        Optional JSON output path.

    Returns
    -------
    results_exp5 : dict[str, Any]
        Part A timings per dataset and Part B timings per training size.
    """
    if datasets_dict is None:
        datasets_dict = prepare_core_datasets()
    D_list = [int(v) for v in D_grid]
    m_list = [int(v) for v in m_scaling_grid]
    if len(D_list) == 0 or any(v <= 0 for v in D_list):
        raise ValueError("D_grid must be a non-empty sequence of positive integers.")
    if len(m_list) == 0 or any(v <= 1 for v in m_list):
        raise ValueError("m_scaling_grid must be a non-empty sequence of integers greater than 1.")
    if D_fixed <= 0 or n_test_scaling <= 0 or num_seeds <= 0 or pegasos_iters <= 0:
        raise ValueError("D_fixed, n_test_scaling, num_seeds and pegasos_iters must be strictly positive.")

    results: dict[str, Any] = {
        "experiment_id": "Exp5_Runtime_vs_Accuracy",
        "D_grid": D_list,
        "m_scaling_grid": m_list,
        "D_fixed": int(D_fixed),
        "num_seeds": int(num_seeds),
        "pegasos_iters": int(pegasos_iters),
        "part_a": {},
        "part_b": {},
    }

    warm = datasets_dict[next(iter(datasets_dict))]
    _time_exact_and_rff(
        warm["X_train"][:200], warm["y_train"][:200], warm["X_test"][:50], warm["y_test"][:50],
        float(warm["gamma_rbf"]), float(warm["lambda_svm"]),
        [10], 200, 1, base_seed,
    )

    for ds_key, bundle in datasets_dict.items():
        X_tr = np.asarray(bundle["X_train"], dtype=np.float64)
        timings = _time_exact_and_rff(
            X_tr, bundle["y_train"], bundle["X_test"], bundle["y_test"],
            float(bundle["gamma_rbf"]), float(bundle["lambda_svm"]),
            D_list, pegasos_iters, num_seeds, base_seed,
        )
        results["part_a"][ds_key] = {
            "dataset_key": ds_key,
            "display_name": bundle.get("display_name", ds_key),
            "d": int(X_tr.shape[1]),
            "m_train": int(X_tr.shape[0]),
            "D_grid": D_list,
            **timings,
        }

    syn = datasets_dict.get("Synthetic_d2")
    if syn is not None:
        gamma = float(syn["gamma_rbf"])
        meta = syn.get("metadata", {})
        per_m: list[dict[str, Any]] = []
        for m_train in m_list:
            X_all, y_all, _ = generate_synthetic_concentric_rings(
                m_total=m_train + n_test_scaling,
                r_0=float(meta.get("r_0", 1.436)),
                sigma_noise=float(meta.get("sigma_noise", 0.35)),
                box_half_width=float(meta.get("box_half_width", 1.8)),
                seed=base_seed + m_train,
            )
            tm = _time_exact_and_rff(
                X_all[:m_train], y_all[:m_train], X_all[m_train:], y_all[m_train:],
                gamma, float(syn["lambda_svm"]),
                [D_fixed], pegasos_iters, num_seeds, base_seed,
            )
            per_m.append({"m_train": m_train, **tm})

        def _collect(model: str, kind: str, key: str) -> np.ndarray:
            if kind == "exact":
                return np.asarray([r["exact"][model][key] for r in per_m], dtype=np.float64)
            return np.asarray([r["rff"][model][key][0] for r in per_m], dtype=np.float64)

        part_b: dict[str, Any] = {"m_grid": m_list, "D_fixed": int(D_fixed), "n_test": int(n_test_scaling)}
        for kind in ("exact", "rff"):
            for key in ("train_time_s", "inference_time_s", "test_error"):
                part_b[f"{kind}_pegasos_{key}"] = _collect("pegasos", kind, key)
        d_in = 2
        part_b["exact_model_floats"] = np.asarray([m * (d_in + 1) for m in m_list], dtype=np.float64)
        part_b["rff_model_floats"] = np.full(len(m_list), float(D_fixed * (d_in + 2)))
        results["part_b"] = part_b

    if save_path is not None:
        _save_json_compatible(results, Path(save_path))
    return results


def _save_json_compatible(data: dict[str, Any], path: Path) -> None:
    """
    Serialize a nested dictionary containing NumPy arrays and scalars to a formatted JSON file.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary to serialize.
    path : Path
        Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default_serializer(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable.")

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_default_serializer)
