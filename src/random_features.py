"""Explicit Random Feature Map constructions for shift-invariant kernels."""

from __future__ import annotations

import numpy as np


class RFFCosinePhase:
    """
    Random Fourier Features with random phase shifts (Algorithm 1 of Rahimi & Recht, 2007).

    Parameters
    ----------
    D : int, default=100
        Number of random Fourier features to sample.
    gamma : float, default=0.5
        Kernel bandwidth parameter (gamma for RBF, gamma_1 for Laplacian).
    kernel_type : str, default="rbf"
        Shift-invariant kernel to approximate: "rbf" or "laplacian".
    seed : int, default=42
        Random seed for reproducible frequency and phase sampling.
    """

    def __init__(
        self,
        D: int = 100,
        gamma: float = 0.5,
        kernel_type: str = "rbf",
        seed: int = 42,
    ) -> None:
        if D <= 0:
            raise ValueError(f"D must be strictly positive, got {D}.")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be strictly positive, got {gamma}.")
        if kernel_type not in ("rbf", "laplacian"):
            raise ValueError(f"kernel_type must be 'rbf' or 'laplacian', got {kernel_type!r}.")

        self.D = int(D)
        self.gamma = float(gamma)
        self.kernel_type = kernel_type
        self.seed = int(seed)

        self.d_: int | None = None
        self.W_: np.ndarray | None = None
        self.b_: np.ndarray | None = None
        self.spectral_second_moment_: float | None = None

    def fit(self, X: np.ndarray) -> RFFCosinePhase:
        """
        Sample random frequency matrix W and phase vector b according to Bochner's theorem.

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Input matrix used only to read input dimensionality d = X.shape[1].

        Returns
        -------
        self : RFFCosinePhase
            Fitted feature map instance.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}.")

        d = int(X.shape[1])
        self.d_ = d
        rng = np.random.default_rng(self.seed)

        if self.kernel_type == "rbf":
            std_omega = np.sqrt(2.0 * self.gamma)
            self.W_ = rng.normal(loc=0.0, scale=std_omega, size=(d, self.D)).astype(np.float64)
            self.spectral_second_moment_ = float(2.0 * d * self.gamma)
        else:
            u = rng.uniform(low=-0.5 * np.pi, high=0.5 * np.pi, size=(d, self.D))
            self.W_ = (self.gamma * np.tan(u)).astype(np.float64)
            self.spectral_second_moment_ = float("inf")

        self.b_ = rng.uniform(low=0.0, high=2.0 * np.pi, size=(1, self.D)).astype(np.float64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Map input points X in R^(m x d) to explicit features Z in R^(m x D).

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Input feature matrix.

        Returns
        -------
        Z : np.ndarray of shape (m, D), dtype=float64
            Randomized cosine feature matrix sqrt(2 / D) * cos(X @ W + b).
        """
        if self.W_ is None or self.b_ is None or self.d_ is None:
            raise RuntimeError("RFFCosinePhase must be fitted before calling transform.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.d_:
            raise ValueError(f"Expected 2D input with d={self.d_} columns, got shape {X.shape}.")

        scale = np.sqrt(2.0 / self.D)
        return scale * np.cos((X @ self.W_) + self.b_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the random parameters on X and return the transformed matrix Z.

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Input feature matrix.

        Returns
        -------
        Z : np.ndarray of shape (m, D), dtype=float64
            Randomized cosine feature matrix.
        """
        return self.fit(X).transform(X)


class RFFCosSinPair:
    """
    Phase-free Random Fourier Features using paired [cos(omega^T x), sin(omega^T x)] coordinates.

    Parameters
    ----------
    D : int, default=100
        Total output feature dimension (must be an even integer; uses D/2 frequencies).
    gamma : float, default=0.5
        Kernel bandwidth parameter for the Gaussian RBF or Laplacian kernel.
    kernel_type : str, default="rbf"
        Shift-invariant kernel to approximate: "rbf" or "laplacian".
    seed : int, default=42
        Random seed for reproducible frequency sampling.
    """

    def __init__(
        self,
        D: int = 100,
        gamma: float = 0.5,
        kernel_type: str = "rbf",
        seed: int = 42,
    ) -> None:
        if D <= 0 or (D % 2) != 0:
            raise ValueError(f"D must be a positive even integer for RFFCosSinPair, got {D}.")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be strictly positive, got {gamma}.")
        if kernel_type not in ("rbf", "laplacian"):
            raise ValueError(f"kernel_type must be 'rbf' or 'laplacian', got {kernel_type!r}.")

        self.D = int(D)
        self.D_half = int(D // 2)
        self.gamma = float(gamma)
        self.kernel_type = kernel_type
        self.seed = int(seed)

        self.d_: int | None = None
        self.W_: np.ndarray | None = None
        self.spectral_second_moment_: float | None = None

    def fit(self, X: np.ndarray) -> RFFCosSinPair:
        """
        Sample D/2 random frequency vectors in R^(d x (D/2)) according to Bochner's theorem.

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Input matrix used only to read input dimensionality d = X.shape[1].

        Returns
        -------
        self : RFFCosSinPair
            Fitted feature map instance.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}.")

        d = int(X.shape[1])
        self.d_ = d
        rng = np.random.default_rng(self.seed)

        if self.kernel_type == "rbf":
            std_omega = np.sqrt(2.0 * self.gamma)
            self.W_ = rng.normal(loc=0.0, scale=std_omega, size=(d, self.D_half)).astype(np.float64)
            self.spectral_second_moment_ = float(2.0 * d * self.gamma)
        else:
            u = rng.uniform(low=-0.5 * np.pi, high=0.5 * np.pi, size=(d, self.D_half))
            self.W_ = (self.gamma * np.tan(u)).astype(np.float64)
            self.spectral_second_moment_ = float("inf")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Map input points X in R^(m x d) to paired cos/sin features Z in R^(m x D).

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Input feature matrix.

        Returns
        -------
        Z : np.ndarray of shape (m, D), dtype=float64
            Concatenated feature matrix sqrt(2 / D) * [cos(X @ W), sin(X @ W)].
        """
        if self.W_ is None or self.d_ is None:
            raise RuntimeError("RFFCosSinPair must be fitted before calling transform.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.d_:
            raise ValueError(f"Expected 2D input with d={self.d_} columns, got shape {X.shape}.")

        proj = X @ self.W_
        scale = np.sqrt(1.0 / self.D_half)
        return scale * np.hstack([np.cos(proj), np.sin(proj)])

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the random frequencies on X and return the transformed matrix Z.

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Input feature matrix.

        Returns
        -------
        Z : np.ndarray of shape (m, D), dtype=float64
            Concatenated cos/sin feature matrix.
        """
        return self.fit(X).transform(X)


class OrthogonalRandomFeatures:
    """
    Orthogonal Random Features (ORF) for the Gaussian RBF kernel via Haar-QR and Chi_d scaling.

    Parameters
    ----------
    D : int, default=100
        Number of random Fourier features to generate.
    gamma : float, default=0.5
        Gaussian RBF kernel bandwidth parameter in exp(-gamma * ||x - y||_2^2).
    seed : int, default=42
        Random seed for reproducible orthogonal block sampling.
    """

    def __init__(
        self,
        D: int = 100,
        gamma: float = 0.5,
        seed: int = 42,
    ) -> None:
        if D <= 0:
            raise ValueError(f"D must be strictly positive, got {D}.")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be strictly positive, got {gamma}.")

        self.D = int(D)
        self.gamma = float(gamma)
        self.seed = int(seed)

        self.d_: int | None = None
        self.W_: np.ndarray | None = None
        self.b_: np.ndarray | None = None
        self.spectral_second_moment_: float | None = None

    def fit(self, X: np.ndarray) -> OrthogonalRandomFeatures:
        """
        Sample block-orthogonal frequency matrix W with exact N(0, 2*gamma*I_d) column marginals.

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Input matrix used only to read input dimensionality d = X.shape[1].

        Returns
        -------
        self : OrthogonalRandomFeatures
            Fitted feature map instance.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}.")

        d = int(X.shape[1])
        self.d_ = d
        rng = np.random.default_rng(self.seed)

        num_blocks = int(np.ceil(self.D / d))
        blocks: list[np.ndarray] = []
        scale_omega = np.sqrt(2.0 * self.gamma)

        for _ in range(num_blocks):
            G_dir = rng.normal(loc=0.0, scale=1.0, size=(d, d))
            Q, R = np.linalg.qr(G_dir)
            signs = np.where(np.diag(R) >= 0.0, 1.0, -1.0)
            Q = Q * signs

            G_rad = rng.normal(loc=0.0, scale=1.0, size=(d, d))
            radii = np.linalg.norm(G_rad, axis=0)

            W_block = scale_omega * (Q * radii)
            blocks.append(W_block)

        W_full = np.hstack(blocks)
        self.W_ = W_full[:, : self.D].astype(np.float64)
        self.b_ = rng.uniform(low=0.0, high=2.0 * np.pi, size=(1, self.D)).astype(np.float64)
        self.spectral_second_moment_ = float(2.0 * d * self.gamma)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Map input points X in R^(m x d) to orthogonal random features Z in R^(m x D).

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Input feature matrix.

        Returns
        -------
        Z : np.ndarray of shape (m, D), dtype=float64
            Orthogonal cosine feature matrix sqrt(2 / D) * cos(X @ W + b).
        """
        if self.W_ is None or self.b_ is None or self.d_ is None:
            raise RuntimeError("OrthogonalRandomFeatures must be fitted before calling transform.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.d_:
            raise ValueError(f"Expected 2D input with d={self.d_} columns, got shape {X.shape}.")

        scale = np.sqrt(2.0 / self.D)
        return scale * np.cos((X @ self.W_) + self.b_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the orthogonal frequency blocks on X and return the transformed matrix Z.

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Input feature matrix.

        Returns
        -------
        Z : np.ndarray of shape (m, D), dtype=float64
            Orthogonal cosine feature matrix.
        """
        return self.fit(X).transform(X)


class RandomBinningFeatures:
    """
    Random Binning Features for the Laplacian kernel (Algorithm 2 of Rahimi & Recht, 2007).

    Parameters
    ----------
    P_grids : int, default=50
        Number of independent random rectilinear grids P to superimpose.
    gamma_1 : float, default=0.5
        Laplacian kernel bandwidth parameter in exp(-gamma_1 * ||x - y||_1).
    seed : int, default=42
        Random seed for reproducible Gamma pitch and Uniform shift sampling.
    """

    def __init__(
        self,
        P_grids: int = 50,
        gamma_1: float = 0.5,
        seed: int = 42,
    ) -> None:
        if P_grids <= 0:
            raise ValueError(f"P_grids must be strictly positive, got {P_grids}.")
        if gamma_1 <= 0.0:
            raise ValueError(f"gamma_1 must be strictly positive, got {gamma_1}.")

        self.P_grids = int(P_grids)
        self.gamma_1 = float(gamma_1)
        self.seed = int(seed)

        self.d_: int | None = None
        self.pitches_: np.ndarray | None = None
        self.shifts_: np.ndarray | None = None
        self.bin_tables_: list[dict[bytes, int]] | None = None
        self.grid_offsets_: np.ndarray | None = None
        self.total_D_: int | None = None

    def fit(self, X: np.ndarray) -> RandomBinningFeatures:
        """
        Sample P random grids (Gamma(2, 1/gamma_1) pitches, Uniform shifts) and index occupied training bins.

        Parameters
        ----------
        X : np.ndarray of shape (m_train, d)
            Training feature matrix used to prune unoccupied bins.

        Returns
        -------
        self : RandomBinningFeatures
            Fitted feature map instance.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}.")

        m_train, d = X.shape
        self.d_ = int(d)
        rng = np.random.default_rng(self.seed)

        self.pitches_ = rng.gamma(shape=2.0, scale=1.0 / self.gamma_1, size=(self.P_grids, d)).astype(np.float64)
        u_unit = rng.uniform(low=0.0, high=1.0, size=(self.P_grids, d))
        self.shifts_ = (u_unit * self.pitches_).astype(np.float64)

        self.bin_tables_ = []
        offsets = [0]

        for p in range(self.P_grids):
            coords = np.floor((X - self.shifts_[p]) / self.pitches_[p]).astype(np.int64)
            table_p: dict[bytes, int] = {}
            for i in range(m_train):
                key = coords[i].tobytes()
                if key not in table_p:
                    table_p[key] = len(table_p)
            self.bin_tables_.append(table_p)
            offsets.append(offsets[-1] + len(table_p))

        self.grid_offsets_ = np.asarray(offsets, dtype=np.int64)
        self.total_D_ = int(offsets[-1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Map input points X in R^(m x d) to one-hot grid indicator features Z in R^(m x total_D_).

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Input feature matrix.

        Returns
        -------
        Z : np.ndarray of shape (m, total_D_), dtype=float64
            Binary grid indicator matrix scaled by 1 / sqrt(P_grids).
        """
        if (
            self.pitches_ is None
            or self.shifts_ is None
            or self.bin_tables_ is None
            or self.grid_offsets_ is None
            or self.total_D_ is None
            or self.d_ is None
        ):
            raise RuntimeError("RandomBinningFeatures must be fitted before calling transform.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.d_:
            raise ValueError(f"Expected 2D input with d={self.d_} columns, got shape {X.shape}.")

        m = X.shape[0]
        Z = np.zeros((m, self.total_D_), dtype=np.float64)
        val = 1.0 / np.sqrt(self.P_grids)

        for p in range(self.P_grids):
            coords = np.floor((X - self.shifts_[p]) / self.pitches_[p]).astype(np.int64)
            table_p = self.bin_tables_[p]
            offset_p = int(self.grid_offsets_[p])
            for i in range(m):
                bin_idx = table_p.get(coords[i].tobytes())
                if bin_idx is not None:
                    Z[i, offset_p + bin_idx] = val

        return Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the random grids on X and return the transformed indicator matrix Z.

        Parameters
        ----------
        X : np.ndarray of shape (m, d)
            Training feature matrix.

        Returns
        -------
        Z : np.ndarray of shape (m, total_D_), dtype=float64
            Binary grid indicator matrix.
        """
        return self.fit(X).transform(X)
