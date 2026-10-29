"""Exact shift-invariant kernel Gram matrix evaluators and approximation metrics."""

from __future__ import annotations

import numpy as np


def gamma_notes_to_paper(gamma_notes: float) -> float:
    """
    Convert kernel bandwidth from lecture notes convention to Rahimi & Recht convention.

    Parameters
    ----------
    gamma_notes : float
        Denominator bandwidth parameter in K(x, x') = exp(-||x - x'||_2^2 / (2 * gamma_notes)).

    Returns
    -------
    gamma_paper : float
        Numerator bandwidth parameter in K(x, x') = exp(-gamma_paper * ||x - x'||_2^2).
    """
    if gamma_notes <= 0.0:
        raise ValueError(f"gamma_notes must be strictly positive, got {gamma_notes}.")
    return float(1.0 / (2.0 * gamma_notes))


def gamma_paper_to_notes(gamma_paper: float) -> float:
    """
    Convert kernel bandwidth from Rahimi & Recht convention to lecture notes convention.

    Parameters
    ----------
    gamma_paper : float
        Numerator bandwidth parameter in K(x, x') = exp(-gamma_paper * ||x - x'||_2^2).

    Returns
    -------
    gamma_notes : float
        Denominator bandwidth parameter in K(x, x') = exp(-||x - x'||_2^2 / (2 * gamma_notes)).
    """
    if gamma_paper <= 0.0:
        raise ValueError(f"gamma_paper must be strictly positive, got {gamma_paper}.")
    return float(1.0 / (2.0 * gamma_paper))


def compute_rbf_gram_matrix(
    X1: np.ndarray,
    X2: np.ndarray | None = None,
    gamma: float = 0.5,
) -> np.ndarray:
    """
    Compute the exact Gaussian RBF kernel Gram matrix K_{i,j} = exp(-gamma * ||X1[i] - X2[j]||_2^2).

    Parameters
    ----------
    X1 : np.ndarray of shape (m1, d)
        First feature matrix.
    X2 : np.ndarray of shape (m2, d) or None, default=None
        Second feature matrix. If None, computes the symmetric self-Gram matrix K(X1, X1).
    gamma : float, default=0.5
        Positive bandwidth parameter in exp(-gamma * ||x - x'||_2^2).

    Returns
    -------
    K : np.ndarray of shape (m1, m2), dtype=float64
        Pairwise Gaussian RBF kernel matrix.
    """
    if gamma <= 0.0:
        raise ValueError(f"gamma must be strictly positive, got {gamma}.")

    X1 = np.asarray(X1, dtype=np.float64)
    if X1.ndim != 2:
        raise ValueError(f"X1 must be a 2D array, got shape {X1.shape}.")

    is_self_gram = X2 is None or X2 is X1
    if is_self_gram:
        X2_arr = X1
    else:
        X2_arr = np.asarray(X2, dtype=np.float64)
        if X2_arr.ndim != 2 or X2_arr.shape[1] != X1.shape[1]:
            raise ValueError(f"Incompatible shapes: X1 {X1.shape} vs X2 {X2_arr.shape}.")

    sq_norms_1 = np.sum(X1 ** 2, axis=1, keepdims=True)
    sq_norms_2 = sq_norms_1.T if is_self_gram else np.sum(X2_arr ** 2, axis=1, keepdims=True).T

    sq_dists = sq_norms_1 + sq_norms_2 - 2.0 * (X1 @ X2_arr.T)
    np.maximum(sq_dists, 0.0, out=sq_dists)

    if is_self_gram:
        np.fill_diagonal(sq_dists, 0.0)

    K = np.exp(-gamma * sq_dists)
    if is_self_gram:
        K = 0.5 * (K + K.T)
    return K


def compute_laplacian_gram_matrix(
    X1: np.ndarray,
    X2: np.ndarray | None = None,
    gamma_1: float = 0.5,
) -> np.ndarray:
    """
    Compute the exact Laplacian kernel Gram matrix K_{i,j} = exp(-gamma_1 * ||X1[i] - X2[j]||_1).

    Parameters
    ----------
    X1 : np.ndarray of shape (m1, d)
        First feature matrix.
    X2 : np.ndarray of shape (m2, d) or None, default=None
        Second feature matrix. If None, computes the symmetric self-Gram matrix K(X1, X1).
    gamma_1 : float, default=0.5
        Positive L1 bandwidth parameter in exp(-gamma_1 * ||x - x'||_1).

    Returns
    -------
    K : np.ndarray of shape (m1, m2), dtype=float64
        Pairwise Laplacian kernel matrix.
    """
    if gamma_1 <= 0.0:
        raise ValueError(f"gamma_1 must be strictly positive, got {gamma_1}.")

    X1 = np.asarray(X1, dtype=np.float64)
    if X1.ndim != 2:
        raise ValueError(f"X1 must be a 2D array, got shape {X1.shape}.")

    is_self_gram = X2 is None or X2 is X1
    if is_self_gram:
        X2_arr = X1
    else:
        X2_arr = np.asarray(X2, dtype=np.float64)
        if X2_arr.ndim != 2 or X2_arr.shape[1] != X1.shape[1]:
            raise ValueError(f"Incompatible shapes: X1 {X1.shape} vs X2 {X2_arr.shape}.")

    m1, d = X1.shape
    m2 = X2_arr.shape[0]
    l1_dists = np.zeros((m1, m2), dtype=np.float64)

    for k in range(d):
        l1_dists += np.abs(X1[:, k : k + 1] - X2_arr[:, k : k + 1].T)

    if is_self_gram:
        np.fill_diagonal(l1_dists, 0.0)

    K = np.exp(-gamma_1 * l1_dists)
    if is_self_gram:
        K = 0.5 * (K + K.T)
    return K


def kernel_matrix_approximation_errors(
    K_exact: np.ndarray,
    K_approx: np.ndarray,
) -> dict[str, float]:
    """
    Compute Frobenius and supremum (Claim 1) approximation errors between two Gram matrices.

    Parameters
    ----------
    K_exact : np.ndarray of shape (m1, m2)
        Ground-truth exact kernel matrix.
    K_approx : np.ndarray of shape (m1, m2)
        Approximate kernel matrix Z1 @ Z2.T induced by a random feature map.

    Returns
    -------
    errors : dict[str, float]
        Dictionary with keys 'frobenius_relative', 'frobenius_absolute',
        'max_abs_error' (sup_{i,j} |K_exact - K_approx|), and 'mean_abs_error'.
    """
    K_exact = np.asarray(K_exact, dtype=np.float64)
    K_approx = np.asarray(K_approx, dtype=np.float64)
    if K_exact.shape != K_approx.shape:
        raise ValueError(f"Shape mismatch: {K_exact.shape} vs {K_approx.shape}.")

    diff = K_exact - K_approx
    abs_diff = np.abs(diff)
    frob_abs = float(np.linalg.norm(diff, ord="fro"))
    frob_exact = float(np.linalg.norm(K_exact, ord="fro"))
    frob_rel = float(frob_abs / max(frob_exact, 1e-15))

    return {
        "frobenius_relative": frob_rel,
        "frobenius_absolute": frob_abs,
        "max_abs_error": float(np.max(abs_diff)),
        "mean_abs_error": float(np.mean(abs_diff)),
    }
