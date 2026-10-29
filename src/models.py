"""Primal and Kernelized Pegasos SVM Classification solvers from scratch."""

import numpy as np

from src.kernels import compute_laplacian_gram_matrix, compute_rbf_gram_matrix


class PrimalPegasosSVM:
    """Primal Pegasos stochastic subgradient solver for L2-regularized Hinge-loss SVM."""

    def __init__(
        self,
        lambda_reg: float = 1e-3,
        n_iters: int = 10000,
        fit_intercept: bool = True,
        seed: int = 42,
    ) -> None:
        """Initialize the Primal Pegasos SVM classifier.

        Parameters
        ----------
        lambda_reg : float, default=1e-3
            Strictly positive L2 regularization parameter lambda > 0.
        n_iters : int, default=10000
            Number of stochastic subgradient iterations T >= 1.
        fit_intercept : bool, default=True
            Whether to update an L2-regularized bias term b alongside w.
        seed : int, default=42
            Random seed for uniform sampling of training indices.
        """
        if lambda_reg <= 0.0:
            raise ValueError(f"lambda_reg must be strictly positive, got {lambda_reg}.")
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}.")

        self.lambda_reg = float(lambda_reg)
        self.n_iters = int(n_iters)
        self.fit_intercept = bool(fit_intercept)
        self.seed = int(seed)

        self.w_bar_: np.ndarray | None = None
        self.b_bar_: float = 0.0
        self.w_last_: np.ndarray | None = None
        self.b_last_: float = 0.0
        self.n_updates_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PrimalPegasosSVM":
        """Fit the Primal Pegasos SVM on feature matrix X and binary labels y.

        Parameters
        ----------
        X : np.ndarray of shape (m, p)
            Training feature matrix (raw features in R^d or random features in R^D).
        y : np.ndarray of shape (m,)
            Binary target labels strictly in {-1, +1}.

        Returns
        -------
        self : PrimalPegasosSVM
            Fitted classifier with learned weights w_bar_ and intercept b_bar_.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array X, got shape {X.shape}.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"Expected 1D label array y of length {X.shape[0]}, got shape {y.shape}.")
        if not np.all((y == -1.0) | (y == 1.0)):
            raise ValueError("Labels y must belong strictly to {-1, +1}.")

        m, p = X.shape
        rng = np.random.default_rng(self.seed)
        sampled_indices = rng.integers(0, m, size=self.n_iters)

        w = np.zeros(p, dtype=np.float64)
        b = 0.0
        w_bar = np.zeros(p, dtype=np.float64)
        b_bar = 0.0
        n_updates = 0

        for t in range(1, self.n_iters + 1):
            idx = int(sampled_indices[t - 1])
            x_t = X[idx]
            y_t = float(y[idx])
            eta_t = 1.0 / (self.lambda_reg * t)
            margin = y_t * (float(np.dot(w, x_t)) + b)

            if margin < 1.0:
                w = (1.0 - 1.0 / t) * w + (eta_t * y_t) * x_t
                if self.fit_intercept:
                    b = (1.0 - 1.0 / t) * b + eta_t * y_t
                n_updates += 1
            else:
                w = (1.0 - 1.0 / t) * w
                if self.fit_intercept:
                    b = (1.0 - 1.0 / t) * b

            w_bar = w_bar + (w - w_bar) / t
            b_bar = b_bar + (b - b_bar) / t

        self.w_bar_ = w_bar
        self.b_bar_ = float(b_bar)
        self.w_last_ = w
        self.b_last_ = float(b)
        self.n_updates_ = int(n_updates)
        return self

    def decision_function(self, X: np.ndarray, use_averaged: bool = True) -> np.ndarray:
        """Compute signed functional margin scores X @ w + b.

        Parameters
        ----------
        X : np.ndarray of shape (m_new, p)
            Input feature matrix.
        use_averaged : bool, default=True
            If True, use Polyak-Ruppert averaged weights (w_bar_, b_bar_); otherwise use final iterate.

        Returns
        -------
        scores : np.ndarray of shape (m_new,)
            Signed functional scores in R.
        """
        if self.w_bar_ is None or self.w_last_ is None:
            raise RuntimeError("PrimalPegasosSVM must be fitted before calling decision_function.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.w_bar_.shape[0]:
            raise ValueError(f"Expected 2D array with {self.w_bar_.shape[0]} features, got shape {X.shape}.")

        w = self.w_bar_ if use_averaged else self.w_last_
        b = self.b_bar_ if use_averaged else self.b_last_
        return X @ w + b

    def predict(self, X: np.ndarray, use_averaged: bool = True) -> np.ndarray:
        """Predict binary class labels in {-1, +1}.

        Parameters
        ----------
        X : np.ndarray of shape (m_new, p)
            Input feature matrix.
        use_averaged : bool, default=True
            If True, use Polyak-Ruppert averaged weights; otherwise use final iterate.

        Returns
        -------
        y_pred : np.ndarray of shape (m_new,)
            Predicted binary labels in {-1.0, +1.0}.
        """
        scores = self.decision_function(X, use_averaged=use_averaged)
        return np.where(scores >= 0.0, 1.0, -1.0)

    def objective_value(self, X: np.ndarray, y: np.ndarray, use_averaged: bool = True) -> float:
        """Evaluate the primal regularized Hinge loss objective F(w, b).

        Parameters
        ----------
        X : np.ndarray of shape (m, p)
            Feature matrix on which to evaluate the objective.
        y : np.ndarray of shape (m,)
            Binary target labels in {-1, +1}.
        use_averaged : bool, default=True
            If True, evaluate at (w_bar_, b_bar_); otherwise evaluate at (w_last_, b_last_).

        Returns
        -------
        obj : float
            Value of (1/m) * sum_i max(0, 1 - y_i * (w^T x_i + b)) + (lambda / 2) * (||w||_2^2 + b^2).
        """
        if self.w_bar_ is None or self.w_last_ is None:
            raise RuntimeError("PrimalPegasosSVM must be fitted before calling objective_value.")
        y = np.asarray(y, dtype=np.float64)
        scores = self.decision_function(X, use_averaged=use_averaged)
        w = self.w_bar_ if use_averaged else self.w_last_
        b = self.b_bar_ if use_averaged else self.b_last_
        hinge_losses = np.maximum(0.0, 1.0 - y * scores)
        norm_sq = float(np.dot(w, w)) + (b * b if self.fit_intercept else 0.0)
        reg_term = 0.5 * self.lambda_reg * norm_sq
        return float(np.mean(hinge_losses) + reg_term)


class KernelPegasosSVM:
    """Kernelized Pegasos stochastic subgradient solver in the Reproducing Kernel Hilbert Space H_K."""

    def __init__(
        self,
        lambda_reg: float = 1e-3,
        gamma: float = 1.0,
        kernel_type: str = "rbf",
        n_iters: int = 10000,
        seed: int = 42,
    ) -> None:
        """Initialize the Kernelized Pegasos SVM classifier.

        Parameters
        ----------
        lambda_reg : float, default=1e-3
            Strictly positive RKHS regularization parameter lambda > 0.
        gamma : float, default=1.0
            Strictly positive bandwidth parameter gamma (RBF) or gamma_1 (Laplacian).
        kernel_type : str, default="rbf"
            Kernel family: "rbf", "laplacian", or "linear".
        n_iters : int, default=10000
            Number of stochastic subgradient iterations T >= 1.
        seed : int, default=42
            Random seed for uniform sampling of training indices.
        """
        if lambda_reg <= 0.0:
            raise ValueError(f"lambda_reg must be strictly positive, got {lambda_reg}.")
        if kernel_type not in ("rbf", "laplacian", "linear"):
            raise ValueError(f"kernel_type must be 'rbf', 'laplacian', or 'linear', got '{kernel_type}'.")
        if kernel_type in ("rbf", "laplacian") and gamma <= 0.0:
            raise ValueError(f"gamma must be strictly positive, got {gamma}.")
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}.")

        self.lambda_reg = float(lambda_reg)
        self.gamma = float(gamma)
        self.kernel_type = kernel_type
        self.n_iters = int(n_iters)
        self.seed = int(seed)

        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.counts_: np.ndarray | None = None
        self.beta_bar_: np.ndarray | None = None
        self.beta_last_: np.ndarray | None = None
        self.support_indices_: np.ndarray | None = None
        self.n_support_: int = 0

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Evaluate the configured kernel matrix K(X1, X2).

        Parameters
        ----------
        X1 : np.ndarray of shape (m1, d)
            First input data matrix.
        X2 : np.ndarray of shape (m2, d)
            Second input data matrix.

        Returns
        -------
        K : np.ndarray of shape (m1, m2)
            Pairwise kernel similarity matrix.
        """
        if self.kernel_type == "rbf":
            return compute_rbf_gram_matrix(X1, X2, gamma=self.gamma)
        if self.kernel_type == "laplacian":
            return compute_laplacian_gram_matrix(X1, X2, gamma_1=self.gamma)
        return np.asarray(X1, dtype=np.float64) @ np.asarray(X2, dtype=np.float64).T

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        K_precomputed: np.ndarray | None = None,
    ) -> "KernelPegasosSVM":
        """Fit the Kernelized Pegasos SVM in RKHS H_K on training data (X, y).

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Training input matrix.
        y : np.ndarray of shape (m,)
            Binary target labels strictly in {-1, +1}.
        K_precomputed : np.ndarray of shape (m, m) or None, default=None
            Optional precomputed m x m training Gram matrix.

        Returns
        -------
        self : KernelPegasosSVM
            Fitted classifier storing dual weights beta_bar_ and support vector indices.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array X, got shape {X.shape}.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"Expected 1D label array y of length {X.shape[0]}, got shape {y.shape}.")
        if not np.all((y == -1.0) | (y == 1.0)):
            raise ValueError("Labels y must belong strictly to {-1, +1}.")

        m = X.shape[0]
        if K_precomputed is not None:
            K_train = np.asarray(K_precomputed, dtype=np.float64)
            if K_train.shape != (m, m):
                raise ValueError(f"Expected K_precomputed of shape {(m, m)}, got {K_train.shape}.")
        else:
            K_train = self._compute_kernel(X, X)

        rng = np.random.default_rng(self.seed)
        sampled_indices = rng.integers(0, m, size=self.n_iters)

        counts = np.zeros(m, dtype=np.int64)
        cy = np.zeros(m, dtype=np.float64)
        beta_bar = np.zeros(m, dtype=np.float64)

        for t in range(1, self.n_iters + 1):
            idx = int(sampled_indices[t - 1])
            y_t = float(y[idx])
            if t == 1:
                g_val = 0.0
            else:
                g_val = float(np.dot(K_train[idx], cy)) / (self.lambda_reg * (t - 1))

            if y_t * g_val < 1.0:
                counts[idx] += 1
                cy[idx] += y_t

            beta_t = cy / (self.lambda_reg * t)
            beta_bar = beta_bar + (beta_t - beta_bar) / t

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.counts_ = counts
        self.beta_bar_ = beta_bar
        self.beta_last_ = cy / (self.lambda_reg * self.n_iters)
        self.support_indices_ = np.flatnonzero(counts > 0)
        self.n_support_ = int(self.support_indices_.size)
        return self

    def decision_function(
        self,
        X: np.ndarray,
        K_cross: np.ndarray | None = None,
        use_averaged: bool = True,
    ) -> np.ndarray:
        """Evaluate the RKHS separator g(x) = sum_{i in S} beta_i K(x, x_i).

        Parameters
        ----------
        X : np.ndarray of shape (m_new, d)
            Query points to score.
        K_cross : np.ndarray of shape (m_new, m) or None, default=None
            Optional precomputed cross-kernel matrix K(X, X_train).
        use_averaged : bool, default=True
            If True, use Polyak-Ruppert averaged dual weights beta_bar_; otherwise use beta_last_.

        Returns
        -------
        scores : np.ndarray of shape (m_new,)
            RKHS functional margin values g(x_j) in R.
        """
        if self.X_train_ is None or self.beta_bar_ is None or self.beta_last_ is None:
            raise RuntimeError("KernelPegasosSVM must be fitted before calling decision_function.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.X_train_.shape[1]:
            raise ValueError(f"Expected 2D array with {self.X_train_.shape[1]} features, got shape {X.shape}.")

        beta = self.beta_bar_ if use_averaged else self.beta_last_

        if K_cross is not None:
            K_mat = np.asarray(K_cross, dtype=np.float64)
            return K_mat @ beta

        if self.n_support_ == 0:
            return np.zeros(X.shape[0], dtype=np.float64)

        sv_idx = self.support_indices_
        K_sv = self._compute_kernel(X, self.X_train_[sv_idx])
        return K_sv @ beta[sv_idx]

    def predict(
        self,
        X: np.ndarray,
        K_cross: np.ndarray | None = None,
        use_averaged: bool = True,
    ) -> np.ndarray:
        """Predict binary class labels in {-1, +1}.

        Parameters
        ----------
        X : np.ndarray of shape (m_new, d)
            Query points to classify.
        K_cross : np.ndarray of shape (m_new, m) or None, default=None
            Optional precomputed cross-kernel matrix K(X, X_train).
        use_averaged : bool, default=True
            If True, use Polyak-Ruppert averaged dual weights; otherwise use final iterate.

        Returns
        -------
        y_pred : np.ndarray of shape (m_new,)
            Predicted binary labels in {-1.0, +1.0}.
        """
        scores = self.decision_function(X, K_cross=K_cross, use_averaged=use_averaged)
        return np.where(scores >= 0.0, 1.0, -1.0)

    def objective_value(
        self,
        K_train: np.ndarray | None = None,
        use_averaged: bool = True,
    ) -> float:
        """Evaluate the RKHS regularized Hinge loss objective F_K(g).

        Parameters
        ----------
        K_train : np.ndarray of shape (m, m) or None, default=None
            Optional precomputed training Gram matrix K(X_train, X_train).
        use_averaged : bool, default=True
            If True, evaluate at beta_bar_; otherwise evaluate at beta_last_.

        Returns
        -------
        obj : float
            Value of (1/m) * sum_i max(0, 1 - y_i g(x_i)) + (lambda / 2) * ||g||_K^2.
        """
        if self.X_train_ is None or self.y_train_ is None or self.beta_bar_ is None or self.beta_last_ is None:
            raise RuntimeError("KernelPegasosSVM must be fitted before calling objective_value.")

        if K_train is None:
            K_mat = self._compute_kernel(self.X_train_, self.X_train_)
        else:
            K_mat = np.asarray(K_train, dtype=np.float64)

        beta = self.beta_bar_ if use_averaged else self.beta_last_
        scores = K_mat @ beta
        hinge_losses = np.maximum(0.0, 1.0 - self.y_train_ * scores)
        rkhs_norm_sq = float(beta @ K_mat @ beta)
        reg_term = 0.5 * self.lambda_reg * rkhs_norm_sq
        return float(np.mean(hinge_losses) + reg_term)
