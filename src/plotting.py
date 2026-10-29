"""Publication-quality Matplotlib figure generators for Core Experiments and Extensions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np


def _apply_publication_style() -> None:
    """
    Configure Matplotlib rcParams for academic publication figures.
    """
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.titlesize": 13,
            "figure.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
        }
    )


def _save_figure_to_paths(
    fig: Figure,
    save_paths: str | Path | Sequence[str | Path] | None,
    dpi: int = 220,
) -> list[Path]:
    """
    Save a Matplotlib figure to one or more file paths, creating parent directories if needed.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save.
    save_paths : str, Path, Sequence[str | Path], or None
        Destination file path(s).
    dpi : int, default=220
        Raster resolution for PNG outputs.

    Returns
    -------
    saved : list[Path]
        List of resolved file paths written to disk.
    """
    if save_paths is None:
        return []
    if isinstance(save_paths, (str, Path)):
        paths_seq = [Path(save_paths)]
    else:
        paths_seq = [Path(p) for p in save_paths]

    saved: list[Path] = []
    for p in paths_seq:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        saved.append(p)
    return saved


def plot_experiment_1_convergence_vs_D(
    results_exp1: dict[str, Any],
    save_paths: str | Path | Sequence[str | Path] | None = None,
    dpi: int = 220,
) -> list[Path]:
    """
    Generate the 2x3 multi-panel convergence figure for Core Experiment 1 across datasets.

    Parameters
    ----------
    results_exp1 : dict[str, Any]
        Output dictionary returned by run_experiment_1_convergence_vs_D().
    save_paths : str, Path, Sequence[str | Path], or None, default=None
        Optional destination path(s) (e.g. PDF for report and PNG for vault).
    dpi : int, default=220
        Resolution in dots per inch for raster outputs.

    Returns
    -------
    saved_paths : list[Path]
        List of saved figure paths.
    """
    _apply_publication_style()
    datasets_map = results_exp1["datasets"]
    ds_keys = list(datasets_map.keys())
    n_rows = len(ds_keys)

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(13.0, 5.2 * n_rows),
        squeeze=False,
    )

    for row_idx, ds_key in enumerate(ds_keys):
        ds = datasets_map[ds_key]
        D_arr = np.asarray(ds["D_grid"], dtype=np.float64)
        display_name = ds["display_name"]
        d_dim = ds["d"]
        gamma = ds["gamma_rbf"]
        bases = ds["baselines"]
        bayes_risk = ds.get("bayes_risk", None)

        ax_gram = axes[row_idx, 0]
        frob_mean = np.asarray(ds["gram_frobenius_rel_mean"], dtype=np.float64)
        frob_std = np.asarray(ds["gram_frobenius_rel_std"], dtype=np.float64)
        sup_mean = np.asarray(ds["gram_max_abs_mean"], dtype=np.float64)
        sup_std = np.asarray(ds["gram_max_abs_std"], dtype=np.float64)

        ax_gram.plot(
            D_arr,
            frob_mean,
            "o-",
            color="#1f77b4",
            lw=2.0,
            ms=5,
            label= (
                r"Rel. Frobenius $\|K - \tilde{K}\|_F / \|K\|_F$"
                f" (slope={ds['empirical_loglog_slope_frobenius']:.2f})"
            ),
        )
        ax_gram.fill_between(
            D_arr,
            np.maximum(frob_mean - frob_std, 1e-6),
            frob_mean + frob_std,
            color="#1f77b4",
            alpha=0.18,
        )

        ax_gram.plot(
            D_arr,
            sup_mean,
            "s-",
            color="#d62728",
            lw=2.0,
            ms=5,
            label=(
                r"Supremum $\|K - \tilde{K}\|_\infty$ (Claim 1)"
                f" (slope={ds['empirical_loglog_slope_supremum']:.2f})"
            ),
        )
        ax_gram.fill_between(
            D_arr,
            np.maximum(sup_mean - sup_std, 1e-6),
            sup_mean + sup_std,
            color="#d62728",
            alpha=0.16,
        )

        ref_anchor = 0.78 * min(float(frob_mean[0]), float(sup_mean[0])) * np.sqrt(D_arr[0])
        ref_curve = ref_anchor / np.sqrt(D_arr)
        ax_gram.plot(
            D_arr,
            ref_curve,
            "k--",
            lw=1.5,
            alpha=0.75,
            label=r"Monte Carlo Rate $\propto D^{-1/2}$ (slope $-0.50$)",
        )

        ax_gram.set_xscale("log")
        ax_gram.set_yscale("log")
        ax_gram.set_xlabel("Number of Random Features $D$ (log scale)")
        ax_gram.set_ylabel("Gram Matrix Approximation Error (log scale)")
        ax_gram.set_title(f"{display_name}\n(A) Kernel Approximation Error vs. $D$ ($\\gamma={gamma}$)")
        ax_gram.legend(loc="upper right", framealpha=0.92)

        ax_svm = axes[row_idx, 1]
        svm_te_mean = np.asarray(ds["rff_svm_test_mean"], dtype=np.float64)
        svm_te_std = np.asarray(ds["rff_svm_test_std"], dtype=np.float64)
        svm_tr_mean = np.asarray(ds["rff_svm_train_mean"], dtype=np.float64)
        svm_tr_std = np.asarray(ds["rff_svm_train_std"], dtype=np.float64)

        ax_svm.plot(
            D_arr,
            svm_te_mean,
            "o-",
            color="#1f77b4",
            lw=2.2,
            ms=5,
            label="RFF + Primal Pegasos (Test Error)",
        )
        ax_svm.fill_between(
            D_arr,
            np.maximum(svm_te_mean - svm_te_std, 0.0),
            np.minimum(svm_te_mean + svm_te_std, 1.0),
            color="#1f77b4",
            alpha=0.20,
        )
        ax_svm.plot(
            D_arr,
            svm_tr_mean,
            "^--",
            color="#2ca02c",
            lw=1.7,
            ms=4.5,
            alpha=0.85,
            label="RFF + Primal Pegasos (Train Error)",
        )
        ax_svm.fill_between(
            D_arr,
            np.maximum(svm_tr_mean - svm_tr_std, 0.0),
            np.minimum(svm_tr_mean + svm_tr_std, 1.0),
            color="#2ca02c",
            alpha=0.12,
        )

        ax_svm.axhline(
            bases["linear_svm_test_error"],
            color="#7f7f7f",
            ls="-.",
            lw=1.8,
            label=f"Linear Pegasos Baseline (Test={bases['linear_svm_test_error']*100:.1f}%)",
        )
        ax_svm.axhline(
            bases["exact_kernel_svm_test_error"],
            color="#d62728",
            ls="--",
            lw=2.0,
            label=f"Exact RBF Kernel Pegasos (Test={bases['exact_kernel_svm_test_error']*100:.2f}%)",
        )
        if bayes_risk is not None:
            oracle_te = ds.get("oracle_test_error", None)
            if oracle_te is not None:
                bayes_lbl = (
                    rf"Bayes Risk ($m\to\infty$: {bayes_risk*100:.2f}%,"
                    f" Sample Test: {oracle_te*100:.2f}%)"
                )
            else:
                bayes_lbl = rf"Asymptotic Bayes Risk ($m\to\infty$: {bayes_risk*100:.2f}%)"
            ax_svm.axhline(
                bayes_risk,
                color="#d4a017",
                ls=":",
                lw=2.2,
                label=bayes_lbl,
            )
        if d_dim > 10 and D_arr[0] < d_dim < D_arr[-1]:
            ax_svm.axvline(
                d_dim,
                color="#9467bd",
                ls=":",
                lw=1.5,
                alpha=0.85,
                label=f"Input Dimension $D = d = {d_dim}$",
            )

        ax_svm.set_xscale("log")
        ax_svm.set_xlabel("Number of Random Features $D$ (log scale)")
        ax_svm.set_ylabel("0-1 Classification Error Rate")
        ax_svm.set_title(f"{display_name}\n(B) Pegasos SVM Classification Error vs. $D$")
        ax_svm.legend(loc="upper right", framealpha=0.92)

    fig.suptitle(
        "Core Experiment 1: RFF Kernel Matrix & Classification Convergence vs. Feature Budget D (Mean ± 1 Std across Seeds)",
        y=1.01,
    )
    fig.tight_layout()
    saved = _save_figure_to_paths(fig, save_paths, dpi=dpi)
    plt.close(fig)
    return saved


def plot_experiment_2_decision_boundaries_2d(
    results_exp2: dict[str, Any],
    save_paths: str | Path | Sequence[str | Path] | None = None,
    max_scatter_points: int = 350,
    seed: int = 42,
    dpi: int = 220,
) -> list[Path]:
    """
    Generate the 2x3 decision boundary and margin gutter progression figure for Core Experiment 2.

    Parameters
    ----------
    results_exp2 : dict[str, Any]
        Output dictionary returned by run_experiment_2_decision_boundaries_2d().
    save_paths : str, Path, Sequence[str | Path], or None, default=None
        Optional destination file path(s).
    max_scatter_points : int, default=350
        Maximum number of test samples overlaid as scatter markers per subplot.
    seed : int, default=42
        Random seed used when subsampling scatter points for visual clarity.
    dpi : int, default=220
        Resolution in dots per inch for raster outputs.

    Returns
    -------
    saved_paths : list[Path]
        List of saved figure paths.
    """
    _apply_publication_style()
    xx = np.asarray(results_exp2["xx"], dtype=np.float64)
    yy = np.asarray(results_exp2["yy"], dtype=np.float64)
    X_test = np.asarray(results_exp2["X_test"], dtype=np.float64)
    y_test = np.asarray(results_exp2["y_test"], dtype=np.float64)
    X_train = np.asarray(results_exp2.get("X_train", X_test), dtype=np.float64)
    y_train = np.asarray(results_exp2.get("y_train", y_test), dtype=np.float64)
    r_0 = float(results_exp2["r_0"])
    bayes_risk = float(results_exp2["bayes_risk"])
    oracle_te = results_exp2.get("oracle_test_error", None)
    if oracle_te is None:
        oracle_te = float(
            np.mean(np.where(np.sum(X_test ** 2, axis=1) >= r_0 ** 2, 1.0, -1.0) != y_test)
        )
    oracle_tr = results_exp2.get("oracle_train_error", None)
    if oracle_tr is None:
        oracle_tr = float(
            np.mean(np.where(np.sum(X_train ** 2, axis=1) >= r_0 ** 2, 1.0, -1.0) != y_train)
        )
    panels = results_exp2["panels"]

    n_panels = len(panels)
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))

    rng = np.random.default_rng(seed)
    if X_test.shape[0] > max_scatter_points:
        sub_idx = rng.choice(X_test.shape[0], size=max_scatter_points, replace=False)
        X_vis = X_test[sub_idx]
        y_vis = y_test[sub_idx]
    else:
        X_vis = X_test
        y_vis = y_test

    neg_mask = y_vis == -1.0
    pos_mask = y_vis == 1.0

    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    circle_x = r_0 * np.cos(theta)
    circle_y = r_0 * np.sin(theta)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15.5, 4.9 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    levels_fill = np.linspace(-2.2, 2.2, 45)

    for idx, panel in enumerate(panels):
        ax = axes_flat[idx]
        score_grid = np.clip(np.asarray(panel["score_grid"], dtype=np.float64), -2.2, 2.2)

        ax.contourf(
            xx,
            yy,
            score_grid,
            levels=levels_fill,
            cmap="RdBu_r",
            alpha=0.68,
            extend="both",
        )

        ax.contour(
            xx,
            yy,
            score_grid,
            levels=[-1.0, 0.0, 1.0],
            colors=["#222222", "#000000", "#222222"],
            linestyles=["--", "-", "--"],
            linewidths=[1.3, 2.3, 1.3],
        )

        ax.plot(
            circle_x,
            circle_y,
            color="#ffd700",
            ls=":",
            lw=2.5,
            label=r"True Bayes Circle $\|x\|_2 = r_0$",
        )

        ax.scatter(
            X_vis[neg_mask, 0],
            X_vis[neg_mask, 1],
            c="#1f77b4",
            edgecolors="k",
            s=22,
            linewidths=0.4,
            alpha=0.82,
            label="Class $-1$ (Inner Disk)",
        )
        ax.scatter(
            X_vis[pos_mask, 0],
            X_vis[pos_mask, 1],
            c="#d62728",
            marker="^",
            edgecolors="k",
            s=24,
            linewidths=0.4,
            alpha=0.82,
            label="Class $+1$ (Outer Ring)",
        )

        te_err = panel["test_error"] * 100.0
        tr_err = panel["train_error"] * 100.0
        rmse_val = panel["grid_score_rmse_vs_exact"]
        subtitle = f"Test Err: {te_err:.2f}% | Train Err: {tr_err:.2f}% | Score RMSE: {rmse_val:.3f}"

        ax.set_title(f"({idx + 1}) {panel['title']}\n{subtitle}", fontsize=10.2)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(float(np.min(xx)), float(np.max(xx)))
        ax.set_ylim(float(np.min(yy)), float(np.max(yy)))
        ax.set_aspect("equal", adjustable="box")

    for extra_idx in range(n_panels, len(axes_flat)):
        axes_flat[extra_idx].axis("off")

    legend_handles = [
        Line2D([0], [0], color="#000000", lw=2.3, ls="-", label=r"Decision Boundary $f(x) = 0$"),
        Line2D([0], [0], color="#222222", lw=1.3, ls="--", label=r"Margin Gutters $f(x) = \pm 1$"),
        Line2D(
            [0],
            [0],
            color="#d4a017",
            lw=2.5,
            ls=":",
            label=(
                rf"Bayes Optimal Circle $\|x\|_2 = {r_0}$ "
                f"($m\\to\\infty$: {bayes_risk*100:.2f}%; "
                f"Sample Test: {oracle_te*100:.2f}%, Train: {oracle_tr*100:.2f}%)"
            ),
        ),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markeredgecolor="k", markersize=6, label="Test Class $-1$"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#d62728", markeredgecolor="k", markersize=6, label="Test Class $+1$"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.02),
        framealpha=0.95,
        fontsize=9.5,
    )

    fig.suptitle(
        "Core Experiment 2: 2D Geometric Progression of Decision Boundaries & Margin Gutters (Synthetic Concentric Rings)",
        y=1.02,
    )
    fig.tight_layout()
    saved = _save_figure_to_paths(fig, save_paths, dpi=dpi)
    plt.close(fig)
    return saved


_D_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]


def plot_experiment_3_bandwidth_effect(
    results_exp3: dict[str, Any],
    save_paths: str | Path | Sequence[str | Path] | None = None,
    dpi: int = 220,
) -> list[Path]:
    """
    Generate the 2x2 bandwidth-sensitivity figure for Extension 1 (rows: error / Gram error; columns: datasets).

    Parameters
    ----------
    results_exp3 : dict[str, Any]
        Output of run_experiment_3_bandwidth_effect().
    save_paths : str, Path, Sequence[str | Path], or None, default=None
        Optional destination path(s).
    dpi : int, default=220
        Raster resolution.

    Returns
    -------
    saved_paths : list[Path]
        Saved figure paths.
    """
    _apply_publication_style()
    ds_keys = list(results_exp3["datasets"].keys())
    fig, axes = plt.subplots(2, len(ds_keys), figsize=(7.6 * len(ds_keys), 10.2), squeeze=False)

    for col, ds_key in enumerate(ds_keys):
        ds = results_exp3["datasets"][ds_key]
        gam = np.asarray(ds["gamma_grid"], dtype=np.float64)
        d_dim = ds["d"]
        D_list = ds["D_subset"]
        ax_e, ax_g = axes[0, col], axes[1, col]

        ax_e.plot(gam, ds["exact_test_error"], "k-o", lw=2.4, ms=5, label="Exact RBF Kernel SVM (Test)")
        ax_e.plot(gam, ds["exact_train_error"], "k--s", lw=1.6, ms=4, alpha=0.7, label="Exact RBF Kernel SVM (Train)")
        for i_d, D in enumerate(D_list):
            c = _D_COLORS[i_d % len(_D_COLORS)]
            mu = np.asarray(ds["rff_test_mean"])[:, i_d]
            sd = np.asarray(ds["rff_test_std"])[:, i_d]
            ax_e.plot(gam, mu, "-^", color=c, lw=1.9, ms=4.5, label=f"RFF + Pegasos, $D={D}$ (Test)")
            ax_e.fill_between(gam, np.maximum(mu - sd, 0.0), np.minimum(mu + sd, 1.0), color=c, alpha=0.16)
        ax_e.axvline(ds["gamma_default"], color="#d4a017", ls=":", lw=2.0, label=r"CV-selected $\gamma^*$")
        ax_e.set_xscale("log")
        ax_e.set_xlabel(r"Bandwidth $\gamma$ (paper convention, log scale)")
        ax_e.set_ylabel("0-1 Classification Error")
        ax_e.set_title(f"{ds['display_name']}\n(A) Error vs. $\\gamma$ (underfit $\\leftarrow$ | $\\rightarrow$ 1-NN overfit)")
        ax_e.legend(loc="upper center", framealpha=0.92, ncol=1)
        sec = ax_e.secondary_xaxis("top", functions=(lambda g, d=d_dim: 2.0 * d * g, lambda s, d=d_dim: s / (2.0 * d)))
        sec.set_xlabel(r"Spectral curvature $\sigma_p^2 = 2d\gamma$")

        for i_d, D in enumerate(D_list):
            c = _D_COLORS[i_d % len(_D_COLORS)]
            mu = np.asarray(ds["gram_frobenius_rel_mean"])[:, i_d]
            sd = np.asarray(ds["gram_frobenius_rel_std"])[:, i_d]
            ax_g.plot(gam, mu, "-o", color=c, lw=2.0, ms=4.5, label=f"$D={D}$")
            ax_g.fill_between(gam, np.maximum(mu - sd, 1e-6), mu + sd, color=c, alpha=0.18)
        ax_g.axvline(ds["gamma_default"], color="#d4a017", ls=":", lw=2.0)
        ax_g.set_xscale("log")
        ax_g.set_yscale("log")
        ax_g.set_xlabel(r"Bandwidth $\gamma$ (paper convention, log scale)")
        ax_g.set_ylabel(r"Relative Frobenius Error $\|K_\gamma - \tilde{K}_\gamma\|_F / \|K_\gamma\|_F$")
        ax_g.set_title(f"{ds['display_name']}\n(B) Kernel Approximation Error vs. $\\gamma$")
        ax_g.legend(loc="upper left", framealpha=0.92)

    fig.suptitle("Extension 1: Effect of the Kernel Bandwidth on the Exact Kernel Machine and its Random Feature Approximation", y=1.0)
    fig.tight_layout()
    saved = _save_figure_to_paths(fig, save_paths, dpi=dpi)
    plt.close(fig)
    return saved


def plot_experiment_4_feature_map_comparison(
    results_exp4: dict[str, Any],
    save_paths: str | Path | Sequence[str | Path] | None = None,
    dpi: int = 220,
) -> list[Path]:
    """
    Generate the feature-map comparison figure for Extension 2 (rows: datasets; columns: RBF/Laplacian x Gram/test error).

    Parameters
    ----------
    results_exp4 : dict[str, Any]
        Output of run_experiment_4_feature_map_comparison().
    save_paths : str, Path, Sequence[str | Path], or None, default=None
        Optional destination path(s).
    dpi : int, default=220
        Raster resolution.

    Returns
    -------
    saved_paths : list[Path]
        Saved figure paths.
    """
    _apply_publication_style()
    ds_keys = list(results_exp4["datasets"].keys())
    fig, axes = plt.subplots(len(ds_keys), 4, figsize=(21.0, 5.0 * len(ds_keys)), squeeze=False)
    rbf_style = {
        "phase_cosine": ("#1f77b4", "o", "Phase-shifted cosine (Alg. 1)"),
        "cos_sin_pair": ("#2ca02c", "s", "Cos-Sin pair"),
        "orthogonal": ("#9467bd", "^", "Orthogonal (ORF)"),
    }
    lap_style = {
        "random_binning": ("#d62728", "D", "Random Binning (Alg. 2)"),
        "cauchy_rff": ("#ff7f0e", "o", "Cauchy RFF (Alg. 1)"),
    }

    for row, ds_key in enumerate(ds_keys):
        ds = results_exp4["datasets"][ds_key]
        D_arr = np.asarray(ds["D_grid"], dtype=np.float64)
        P_arr = np.asarray(ds["P_grid"], dtype=np.float64)

        panels = [
            (axes[row, 0], ds["rbf_maps"], rbf_style, D_arr, "gram_frobenius_rel", "Number of features $D$",
             f"(A) RBF kernel ($\\gamma={ds['gamma_rbf']:.4g}$): Gram error"),
            (axes[row, 1], ds["rbf_maps"], rbf_style, D_arr, "test_error", "Number of features $D$",
             "(B) RBF kernel: Pegasos SVM test error"),
            (axes[row, 2], ds["laplacian_maps"], lap_style, P_arr, "gram_frobenius_rel", "Budget $P$ (grids) $=D$ (features)",
             f"(C) Laplacian kernel ($\\gamma_1={ds['gamma_laplacian']:.4g}$): Gram error"),
            (axes[row, 3], ds["laplacian_maps"], lap_style, P_arr, "test_error", "Budget $P$ (grids) $=D$ (features)",
             "(D) Laplacian kernel: Pegasos SVM test error"),
        ]
        for ax, maps, style, x_arr, metric, xlabel, title in panels:
            for name, (color, marker, label) in style.items():
                mp = maps[name]
                if metric == "gram_frobenius_rel":
                    mu = np.asarray(mp["gram_frobenius_rel_mean"])
                    sd = np.asarray(mp["gram_frobenius_rel_std"])
                    ax.plot(x_arr, mu, marker=marker, color=color, lw=2.0, ms=5, label=label)
                    ax.fill_between(x_arr, np.maximum(mu - sd, 1e-6), mu + sd, color=color, alpha=0.16)
                    ax.set_yscale("log")
                    ax.set_ylabel(r"Relative Frobenius Error")
                else:
                    mu = np.asarray(mp["test_error_mean"])
                    sd = np.asarray(mp["test_error_std"])
                    ax.plot(x_arr, mu, marker=marker, color=color, lw=2.0, ms=5, label=label)
                    ax.fill_between(x_arr, np.maximum(mu - sd, 0.0), np.minimum(mu + sd, 1.0), color=color, alpha=0.16)
                    ax.set_ylabel("0-1 Test Error")
            if metric == "gram_frobenius_rel":
                min_first = min(float(np.asarray(maps[k]["gram_frobenius_rel_mean"])[0]) for k in style)
                ref = 0.78 * min_first * np.sqrt(x_arr[0]) / np.sqrt(x_arr)
                ax.plot(x_arr, ref, "k--", lw=1.3, alpha=0.7, label=r"$\propto D^{-1/2}$ reference (slope $-0.50$)")
            else:
                key = "exact_rbf_test_error" if maps is ds["rbf_maps"] else "exact_laplacian_test_error"
                ax.axhline(ds[key], color="#d62728" if maps is ds["rbf_maps"] else "#8c1c13", ls="--", lw=2.0,
                           label=f"Exact kernel SVM ({ds[key] * 100:.2f}%)")
            ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_title(f"{ds['display_name']}\n{title}", fontsize=10)
            ax.legend(loc="upper right", framealpha=0.92, fontsize=8)

    fig.suptitle("Extension 2: Comparison of Random Feature Map Constructions (Mean $\\pm$ 1 Std across Seeds)", y=1.0)
    fig.tight_layout()
    saved = _save_figure_to_paths(fig, save_paths, dpi=dpi)
    plt.close(fig)
    return saved


def plot_experiment_5_runtime_vs_D(
    results_exp5: dict[str, Any],
    save_paths: str | Path | Sequence[str | Path] | None = None,
    dpi: int = 220,
) -> list[Path]:
    """
    Generate the Part A figure of Extension 3: wall-clock time vs. D and the accuracy/training-time Pareto plot.

    Parameters
    ----------
    results_exp5 : dict[str, Any]
        Output of run_experiment_5_runtime_vs_accuracy().
    save_paths : str, Path, Sequence[str | Path], or None, default=None
        Optional destination path(s).
    dpi : int, default=220
        Raster resolution.

    Returns
    -------
    saved_paths : list[Path]
        Saved figure paths.
    """
    _apply_publication_style()
    ds_keys = list(results_exp5["part_a"].keys())
    fig, axes = plt.subplots(len(ds_keys), 3, figsize=(18.0, 5.2 * len(ds_keys)), squeeze=False)

    for row, ds_key in enumerate(ds_keys):
        ds = results_exp5["part_a"][ds_key]
        D_arr = np.asarray(ds["D_grid"], dtype=np.float64)
        m_tr = ds["m_train"]
        ex, rf = ds["exact"], ds["rff"]
        ax_t, ax_i, ax_p = axes[row, 0], axes[row, 1], axes[row, 2]

        ax_t.plot(D_arr, rf["pegasos"]["train_time_s"] * 1e3, "-", color="#1f77b4", marker="o", ms=4.5, lw=2.0,
                  label="RFF + Primal Pegasos SVM")
        ax_t.axhline(ex["pegasos"]["train_time_s"] * 1e3, color="#d62728", ls="--", lw=1.8,
                     label="Exact Kernel Pegasos SVM")

        ax_i.plot(D_arr, rf["pegasos"]["inference_time_s"] * 1e3, "-", color="#1f77b4", marker="o", ms=4.5, lw=2.0,
                  label="RFF + Primal Pegasos SVM")
        ax_i.axhline(ex["pegasos"]["inference_time_s"] * 1e3, color="#d62728", ls="--", lw=1.8,
                     label="Exact Kernel Pegasos SVM")

        for ax, ttl, yl in ((ax_t, "(A) Training time vs. $D$", "Wall-clock training time (ms)"),
                            (ax_i, "(B) Inference time vs. $D$", "Wall-clock inference time on test set (ms)")):
            ax.axvline(m_tr, color="#d4a017", ls=":", lw=2.0, label=f"$D = m_{{train}} = {m_tr}$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Number of random features $D$")
            ax.set_ylabel(yl)
            ax.set_title(f"{ds['display_name']}\n{ttl}")
            ax.legend(loc="upper left", framealpha=0.92, fontsize=8)

        ax_p.plot(rf["pegasos"]["train_time_s"] * 1e3, rf["pegasos"]["test_error"], marker="o", color="#1f77b4", lw=1.8, ms=5,
                  label="RFF + Primal Pegasos (sweep over $D$)")
        ax_p.scatter([ex["pegasos"]["train_time_s"] * 1e3], [ex["pegasos"]["test_error"]], color="#d62728", marker="*", s=240,
                     edgecolors="k", zorder=5, label="Exact Kernel Pegasos SVM")
        ax_p.set_xscale("log")
        ax_p.set_xlabel("Wall-clock training time (ms)")
        ax_p.set_ylabel("0-1 Test Error")
        ax_p.set_title(f"{ds['display_name']}\n(C) Accuracy vs. training-time Pareto plot")
        ax_p.legend(loc="upper right", framealpha=0.92, fontsize=8)

    fig.suptitle("Extension 3 (Part A): Runtime vs. Accuracy as a Function of $D$ at Fixed Training Size", y=1.0)
    fig.tight_layout()
    saved = _save_figure_to_paths(fig, save_paths, dpi=dpi)
    plt.close(fig)
    return saved


def plot_experiment_5_runtime_vs_m(
    results_exp5: dict[str, Any],
    save_paths: str | Path | Sequence[str | Path] | None = None,
    dpi: int = 220,
) -> list[Path]:
    """
    Generate the Part B figure of Extension 3: wall-clock time and error vs. training size m at fixed D.

    Parameters
    ----------
    results_exp5 : dict[str, Any]
        Output of run_experiment_5_runtime_vs_accuracy() containing a non-empty 'part_b'.
    save_paths : str, Path, Sequence[str | Path], or None, default=None
        Optional destination path(s).
    dpi : int, default=220
        Raster resolution.

    Returns
    -------
    saved_paths : list[Path]
        Saved figure paths.
    """
    _apply_publication_style()
    pb = results_exp5["part_b"]
    if not pb:
        raise ValueError("results_exp5['part_b'] is empty (Synthetic_d2 dataset required).")
    m_arr = np.asarray(pb["m_grid"], dtype=np.float64)
    D_fixed = pb["D_fixed"]
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4))

    axes[0].plot(m_arr, pb["exact_pegasos_train_time_s"] * 1e3, "-", color="#d62728", marker="o", lw=2.0,
                 label="Exact Kernel Pegasos SVM")
    axes[0].plot(m_arr, pb["rff_pegasos_train_time_s"] * 1e3, "-", color="#1f77b4", marker="s", lw=2.0,
                 label=f"RFF ($D={D_fixed}$) + Primal Pegasos")
    axes[1].plot(m_arr, pb["exact_pegasos_inference_time_s"] * 1e3, "-", color="#d62728", marker="o", lw=2.0,
                 label="Exact Kernel Pegasos SVM")
    axes[1].plot(m_arr, pb["rff_pegasos_inference_time_s"] * 1e3, "-", color="#1f77b4", marker="s", lw=2.0,
                 label=f"RFF ($D={D_fixed}$) + Primal Pegasos")
    axes[2].plot(m_arr, pb["exact_pegasos_test_error"], "-", color="#d62728", marker="o", lw=2.0,
                 label="Exact Kernel Pegasos SVM")
    axes[2].plot(m_arr, pb["rff_pegasos_test_error"], "-", color="#1f77b4", marker="s", lw=2.0,
                 label=f"RFF ($D={D_fixed}$) + Primal Pegasos")

    for ax, ttl, yl in ((axes[0], "(A) Training time vs. $m_{train}$", "Wall-clock training time (ms)"),
                        (axes[1], "(B) Inference time vs. $m_{train}$", "Wall-clock inference time (ms)")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training set size $m_{train}$")
        ax.set_ylabel(yl)
        ax.set_title(f"Synthetic Concentric Rings (d=2)\n{ttl}")
        ax.legend(loc="upper left", framealpha=0.92, fontsize=8)
    ref_x = m_arr
    axes[0].plot(ref_x, pb["exact_pegasos_train_time_s"][0] * 1e3 * (ref_x / ref_x[0]) ** 2, "k:", lw=1.2, alpha=0.6, label="$\\propto m^2$ reference")
    axes[0].plot(ref_x, pb["rff_pegasos_train_time_s"][0] * 1e3 * (ref_x / ref_x[0]), "k-.", lw=1.2, alpha=0.6, label="$\\propto m$ reference")
    axes[0].legend(loc="upper left", framealpha=0.92, fontsize=8)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Training set size $m_{train}$")
    axes[2].set_ylabel("0-1 Test Error")
    axes[2].set_title("Synthetic Concentric Rings (d=2)\n(C) Test error vs. $m_{train}$")
    axes[2].legend(loc="upper right", framealpha=0.92, fontsize=8)

    fig.suptitle("Extension 3 (Part B): Runtime Scaling in the Training Set Size at Fixed $D$", y=1.02)
    fig.tight_layout()
    saved = _save_figure_to_paths(fig, save_paths, dpi=dpi)
    plt.close(fig)
    return saved
