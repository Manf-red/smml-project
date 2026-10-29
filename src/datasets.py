"""Synthetic and real-world classification dataset generators, loaders, and zero-leakage preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import ndtr
from sklearn.datasets import load_digits


def generate_synthetic_concentric_rings(
    m_total: int = 1500,
    r_0: float = 1.436,
    sigma_noise: float = 0.35,
    box_half_width: float = 1.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Generate the 2D Noisy Concentric Rings binary classification dataset.

    Parameters
    ----------
    m_total : int, default=1500
        Total number of data points to sample.
    r_0 : float, default=1.436
        Radius of the circular decision boundary ||x||_2 = r_0.
    sigma_noise : float, default=0.35
        Standard deviation of the zero-mean Gaussian latent noise.
    box_half_width : float, default=1.8
        Half-width B of the square sampling domain [-B, B]^2.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (m_total, 2), dtype=float64
        2D coordinates of the sampled points.
    y : np.ndarray of shape (m_total,), dtype=float64
        Binary class labels in {-1.0, +1.0}.
    metadata : dict[str, Any]
        Diagnostic metrics (bayes_risk, oracle_empirical_error, infiltration counts).
    """
    if m_total <= 0:
        raise ValueError(f"m_total must be strictly positive, got {m_total}.")
    if r_0 <= 0.0:
        raise ValueError(f"r_0 must be strictly positive, got {r_0}.")
    if sigma_noise < 0.0:
        raise ValueError(f"sigma_noise must be non-negative, got {sigma_noise}.")
    if box_half_width <= 0.0:
        raise ValueError(f"box_half_width must be strictly positive, got {box_half_width}.")

    rng = np.random.default_rng(seed)
    X = rng.uniform(low=-box_half_width, high=box_half_width, size=(m_total, 2)).astype(np.float64)

    squared_norms = np.sum(X ** 2, axis=1)
    f_star = squared_norms - (r_0 ** 2)

    if sigma_noise > 0.0:
        epsilon = rng.normal(loc=0.0, scale=sigma_noise, size=m_total)
        pointwise_bayes_error = ndtr(-np.abs(f_star) / sigma_noise)
        bayes_risk = float(np.mean(pointwise_bayes_error))
    else:
        epsilon = np.zeros(m_total, dtype=np.float64)
        bayes_risk = 0.0

    noisy_latent_score = f_star + epsilon
    y = np.where(noisy_latent_score >= 0.0, 1.0, -1.0).astype(np.float64)
    y_clean_oracle = np.where(f_star >= 0.0, 1.0, -1.0).astype(np.float64)

    metadata: dict[str, Any] = {
        "dataset_name": "Synthetic Concentric Rings (d=2)",
        "m_total": m_total,
        "d": 2,
        "r_0": r_0,
        "sigma_noise": sigma_noise,
        "box_half_width": box_half_width,
        "diameter_bound": float(np.sqrt(2.0) * (2.0 * box_half_width)),
        "bayes_risk": bayes_risk,
        "oracle_empirical_error": float(np.mean(y != y_clean_oracle)),
        "infiltrated_pos_inside_circle": int(np.sum((f_star < 0.0) & (y == 1.0))),
        "infiltrated_neg_outside_circle": int(np.sum((f_star >= 0.0) & (y == -1.0))),
        "pos_class_fraction": float(np.mean(y == 1.0)),
        "f_star": f_star,
    }
    return X, y, metadata


def load_uci_digits_even_odd() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load the UCI Handwritten Digits dataset mapped to Even (-1.0) vs. Odd (+1.0).

    Returns
    -------
    X : np.ndarray of shape (1797, 64), dtype=float64
        Flattened 8x8 pixel intensity vectors in [0.0, 16.0].
    y : np.ndarray of shape (1797,), dtype=float64
        Binary parity labels (-1.0 for Even digits, +1.0 for Odd digits).
    metadata : dict[str, Any]
        Dataset shape, class counts, pixel bounds, and raw integer digit labels.
    """
    X_raw, raw_targets_raw = load_digits(return_X_y=True)
    X = np.asarray(X_raw, dtype=np.float64)
    raw_targets = np.asarray(raw_targets_raw, dtype=np.int64)

    y = np.where((raw_targets % 2) == 1, 1.0, -1.0).astype(np.float64)

    m_total, d = X.shape
    num_even = int(np.sum(y == -1.0))
    num_odd = int(np.sum(y == 1.0))

    metadata: dict[str, Any] = {
        "dataset_name": "UCI Handwritten Digits Even-vs-Odd (d=64)",
        "m_total": m_total,
        "d": d,
        "num_even_neg": num_even,
        "num_odd_pos": num_odd,
        "pos_class_fraction": float(num_odd / m_total),
        "pixel_min": float(np.min(X)),
        "pixel_max": float(np.max(X)),
        "raw_digit_targets": raw_targets,
    }
    return X, y, metadata


def standardize_features_zero_leakage(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize train and evaluation matrices using exclusively training statistics.

    Parameters
    ----------
    X_train : np.ndarray of shape (m_train, d)
        Training feature matrix used to estimate feature means and standard deviations.
    X_eval : np.ndarray of shape (m_eval, d)
        Evaluation feature matrix transformed using X_train statistics.
    eps : float, default=1e-12
        Numerical threshold below which a feature is treated as constant (sigma set to 1.0).

    Returns
    -------
    X_train_std : np.ndarray of shape (m_train, d), dtype=float64
        Standardized training matrix.
    X_eval_std : np.ndarray of shape (m_eval, d), dtype=float64
        Standardized evaluation matrix.
    mu : np.ndarray of shape (d,), dtype=float64
        Feature-wise training means.
    sigma_safe : np.ndarray of shape (d,), dtype=float64
        Feature-wise training standard deviations with constant-feature guard applied.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_eval = np.asarray(X_eval, dtype=np.float64)
    if X_train.ndim != 2 or X_eval.ndim != 2:
        raise ValueError("X_train and X_eval must be 2D arrays.")
    if X_train.shape[1] != X_eval.shape[1]:
        raise ValueError(f"Feature dimension mismatch: {X_train.shape[1]} vs {X_eval.shape[1]}.")

    mu = np.mean(X_train, axis=0)
    sigma_raw = np.std(X_train, axis=0)
    sigma = np.where(sigma_raw < eps, 1.0, sigma_raw)

    X_train_std = (X_train - mu) / sigma
    X_eval_std = (X_eval - mu) / sigma
    return X_train_std, X_eval_std, mu, sigma


def prepare_dataset_zero_leakage(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.30,
    standardize: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Perform a stratified train/test split followed by optional zero-leakage standardization.

    Parameters
    ----------
    X : np.ndarray of shape (m_total, d)
        Full feature matrix.
    y : np.ndarray of shape (m_total,)
        Full binary label vector in {-1.0, +1.0}.
    test_ratio : float, default=0.30
        Fraction of samples from each class allocated to the held-out test split.
    standardize : bool, default=True
        Whether to apply training-only z-score standardization after splitting.
    seed : int, default=42
        Random seed for the stratified permutation.

    Returns
    -------
    X_train : np.ndarray of shape (m_train, d), dtype=float64
        Training feature matrix.
    X_test : np.ndarray of shape (m_test, d), dtype=float64
        Test feature matrix.
    y_train : np.ndarray of shape (m_train,), dtype=float64
        Training binary labels.
    y_test : np.ndarray of shape (m_test,), dtype=float64
        Test binary labels.
    split_info : dict[str, Any]
        Diagnostic dictionary with split sizes, indices, and standardization parameters.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Sample count mismatch: X has {X.shape[0]}, y has {y.shape[0]}.")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}.")

    rng = np.random.default_rng(seed)
    neg_indices = rng.permutation(np.flatnonzero(y == -1.0))
    pos_indices = rng.permutation(np.flatnonzero(y == 1.0))

    n_test_neg = int(np.round(len(neg_indices) * test_ratio))
    n_test_pos = int(np.round(len(pos_indices) * test_ratio))

    test_idx = rng.permutation(np.concatenate([neg_indices[:n_test_neg], pos_indices[:n_test_pos]]))
    train_idx = rng.permutation(np.concatenate([neg_indices[n_test_neg:], pos_indices[n_test_pos:]]))

    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if standardize:
        X_train, X_test, mu, sigma_safe = standardize_features_zero_leakage(X_train_raw, X_test_raw)
        num_constant_features = int(np.sum( np.std(X_train_raw, axis=0) < 1e-12 ))
    else:
        X_train, X_test = X_train_raw.copy(), X_test_raw.copy()
        mu = np.zeros(X.shape[1], dtype=np.float64)
        sigma_safe = np.ones(X.shape[1], dtype=np.float64)
        num_constant_features = 0

    split_info: dict[str, Any] = {
        "m_train": int(X_train.shape[0]),
        "m_test": int(X_test.shape[0]),
        "d": int(X.shape[1]),
        "test_ratio": test_ratio,
        "standardize": standardize,
        "train_pos_fraction": float(np.mean(y_train == 1.0)),
        "test_pos_fraction": float(np.mean(y_test == 1.0)),
        "num_constant_features": num_constant_features,
        "mu_train": mu,
        "sigma_train_safe": sigma_safe,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }
    return X_train, X_test, y_train, y_test, split_info


def stratified_kfold_indices(
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate stratified K-Fold cross-validation (train_idx, val_idx) index pairs.

    Parameters
    ----------
    y : np.ndarray of shape (m,)
        Binary label vector in {-1.0, +1.0} to stratify across folds.
    n_splits : int, default=5
        Number of cross-validation folds K.
    seed : int, default=42
        Random seed for shuffling class indices before fold partitioning.

    Returns
    -------
    folds : list[tuple[np.ndarray, np.ndarray]]
        List of K tuples (fold_train_idx, fold_val_idx).
    """
    y = np.asarray(y, dtype=np.float64)
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}.")

    rng = np.random.default_rng(seed)
    neg_indices = rng.permutation(np.flatnonzero(y == -1.0))
    pos_indices = rng.permutation(np.flatnonzero(y == 1.0))

    if len(neg_indices) < n_splits or len(pos_indices) < n_splits:
        raise ValueError("Each class must have at least n_splits samples.")

    neg_folds = np.array_split(neg_indices, n_splits)
    pos_folds = np.array_split(pos_indices, n_splits)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        val_idx = rng.permutation(np.concatenate([neg_folds[k], pos_folds[k]]))
        train_parts = [
            np.concatenate([neg_folds[j], pos_folds[j]])
            for j in range(n_splits)
            if j != k
        ]
        train_idx = rng.permutation(np.concatenate(train_parts))
        folds.append((train_idx, val_idx))

    return folds
