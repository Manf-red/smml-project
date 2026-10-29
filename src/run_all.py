"""Single entry point reproducing every experiment, result file, and figure of the project."""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import (
    _save_json_compatible,
    prepare_core_datasets,
    run_experiment_1_convergence_vs_D,
    run_experiment_2_decision_boundaries_2d,
    run_experiment_3_bandwidth_effect,
    run_experiment_4_feature_map_comparison,
    run_experiment_5_runtime_vs_accuracy,
    select_hyperparameters_by_cv,
)
from src.plotting import (
    plot_experiment_1_convergence_vs_D,
    plot_experiment_2_decision_boundaries_2d,
    plot_experiment_3_bandwidth_effect,
    plot_experiment_4_feature_map_comparison,
    plot_experiment_5_runtime_vs_D,
    plot_experiment_5_runtime_vs_m,
)

FULL_CONFIG: dict[str, Any] = {
    "m_synthetic": 1500,
    "cv_splits": 5,
    "cv_pegasos_iters": 20000,
    "num_seeds": 10,
    "pegasos_iters": 20000,
    "exp1_D_grid": (5, 10, 25, 50, 100, 250, 500, 1000, 2000),
    "exp2_D_list": (5, 15, 50, 500),
    "exp2_grid_resolution": 200,
    "exp3_D_subset": (25, 100, 500),
    "exp4_D_grid": (10, 20, 50, 100, 250, 500, 1000),
    "exp4_P_grid": (1, 2, 5, 10, 20, 50, 100, 200),
    "exp5_D_grid": (5, 10, 25, 50, 100, 250, 500, 1000, 2000),
    "exp5_m_grid": (250, 500, 1000, 2000, 4000, 6000),
    "exp5_num_seeds": 3,
    "exp5_pegasos_iters": 10000,
}

QUICK_CONFIG: dict[str, Any] = {
    "m_synthetic": 600,
    "cv_splits": 3,
    "cv_pegasos_iters": 3000,
    "num_seeds": 2,
    "pegasos_iters": 3000,
    "exp1_D_grid": (10, 50, 200),
    "exp2_D_list": (5, 50),
    "exp2_grid_resolution": 40,
    "exp3_D_subset": (25, 100),
    "exp4_D_grid": (10, 50, 100),
    "exp4_P_grid": (2, 10, 50),
    "exp5_D_grid": (10, 100, 500),
    "exp5_m_grid": (250, 500, 1000),
    "exp5_num_seeds": 1,
    "exp5_pegasos_iters": 2000,
}


def run_pipeline(
    quick: bool = False,
    seed: int = 42,
    results_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    vault_assets_dir: str | Path | None = None,
    use_cv: bool = True,
) -> dict[str, Any]:
    """
    Run hyperparameter selection, Experiments 1-5, and figure generation end to end.

    Parameters
    ----------
    quick : bool, default=False
        If True, use the reduced QUICK_CONFIG (smoke test); otherwise the full FULL_CONFIG.
    seed : int, default=42
        Master seed for dataset synthesis, splitting, and cross-validation.
    results_dir : str, Path, or None, default=None
        Output directory for JSON results (default: <project>/results/final).
    figures_dir : str, Path, or None, default=None
        Output directory for PDF figures (default: <project>/report/figures).
    vault_assets_dir : str, Path, or None, default=None
        Optional directory for PNG copies of the figures (None disables PNG export).
    use_cv : bool, default=True
        If True, select (gamma, lambda) by cross-validation on the training splits;
        otherwise keep the defaults of prepare_core_datasets().

    Returns
    -------
    summary : dict[str, Any]
        Configuration, selected hyperparameters, per-stage wall-clock seconds, and output file paths.
    """
    cfg = dict(QUICK_CONFIG if quick else FULL_CONFIG)
    res_dir = Path(results_dir) if results_dir is not None else PROJECT_ROOT / "results" / "final"
    fig_dir = Path(figures_dir) if figures_dir is not None else PROJECT_ROOT / "report" / "figures"
    png_dir = Path(vault_assets_dir) if vault_assets_dir is not None else None
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)

    def _fig_paths(pdf_name: str, png_name: str) -> list[Path]:
        paths = [fig_dir / pdf_name]
        if png_dir is not None:
            paths.append(png_dir / png_name)
        return paths

    stage_seconds: dict[str, float] = {}
    outputs: dict[str, list[str]] = {"results": [], "figures": []}

    def _stage(name: str) -> float:
        print(f"[run_all] {name} ...", flush=True)
        return time.perf_counter()

    def _done(name: str, t0: float) -> None:
        stage_seconds[name] = time.perf_counter() - t0
        print(f"[run_all]   done in {stage_seconds[name]:.1f}s", flush=True)

    t0 = _stage("datasets")
    datasets = prepare_core_datasets(m_synthetic=cfg["m_synthetic"], test_ratio=0.30, seed=seed)
    _done("datasets", t0)

    selection_report: dict[str, Any] = {}
    if use_cv:
        t0 = _stage("cross_validation")
        datasets, selection_report = select_hyperparameters_by_cv(
            datasets, n_splits=cfg["cv_splits"], pegasos_iters=cfg["cv_pegasos_iters"], seed=seed
        )
        path = res_dir / "hyperparameter_selection.json"
        _save_json_compatible(selection_report, path)
        outputs["results"].append(str(path))
        _done("cross_validation", t0)

    t0 = _stage("experiment_1")
    res1 = run_experiment_1_convergence_vs_D(
        datasets, D_grid=cfg["exp1_D_grid"], num_seeds=cfg["num_seeds"],
        pegasos_iters=cfg["pegasos_iters"], save_path=res_dir / "exp1_convergence_vs_D.json",
    )
    outputs["results"].append(str(res_dir / "exp1_convergence_vs_D.json"))
    outputs["figures"] += [str(p) for p in plot_experiment_1_convergence_vs_D(
        res1, _fig_paths("fig1_convergence_vs_D.pdf", "fig_final_exp1_convergence_vs_D.png"))]
    _done("experiment_1", t0)

    t0 = _stage("experiment_2")
    res2 = run_experiment_2_decision_boundaries_2d(
        datasets["Synthetic_d2"], D_list=cfg["exp2_D_list"], grid_resolution=cfg["exp2_grid_resolution"],
        pegasos_iters=cfg["pegasos_iters"], seed=seed, save_path=res_dir / "exp2_decision_boundaries_2d.json",
    )
    outputs["results"].append(str(res_dir / "exp2_decision_boundaries_2d.json"))
    outputs["figures"] += [str(p) for p in plot_experiment_2_decision_boundaries_2d(
        res2, _fig_paths("fig2_decision_boundaries_2d.pdf", "fig_final_exp2_decision_boundaries_2d.png"))]
    _done("experiment_2", t0)

    t0 = _stage("experiment_3")
    res3 = run_experiment_3_bandwidth_effect(
        datasets, D_subset=cfg["exp3_D_subset"], num_seeds=cfg["num_seeds"],
        save_path=res_dir / "exp3_bandwidth_effect.json",
    )
    outputs["results"].append(str(res_dir / "exp3_bandwidth_effect.json"))
    outputs["figures"] += [str(p) for p in plot_experiment_3_bandwidth_effect(
        res3, _fig_paths("fig3_bandwidth_gamma_effect.pdf", "fig_step9_exp3_bandwidth_effect.png"))]
    _done("experiment_3", t0)

    t0 = _stage("experiment_4")
    res4 = run_experiment_4_feature_map_comparison(
        datasets, D_grid=cfg["exp4_D_grid"], P_grid=cfg["exp4_P_grid"], num_seeds=cfg["num_seeds"],
        save_path=res_dir / "exp4_feature_map_comparison.json",
    )
    outputs["results"].append(str(res_dir / "exp4_feature_map_comparison.json"))
    outputs["figures"] += [str(p) for p in plot_experiment_4_feature_map_comparison(
        res4, _fig_paths("fig4_feature_map_constructions.pdf", "fig_step9_exp4_feature_map_constructions.png"))]
    _done("experiment_4", t0)

    t0 = _stage("experiment_5")
    res5 = run_experiment_5_runtime_vs_accuracy(
        datasets, D_grid=cfg["exp5_D_grid"], m_scaling_grid=cfg["exp5_m_grid"], D_fixed=300,
        num_seeds=cfg["exp5_num_seeds"], pegasos_iters=cfg["exp5_pegasos_iters"],
        save_path=res_dir / "exp5_runtime_vs_accuracy.json",
    )
    outputs["results"].append(str(res_dir / "exp5_runtime_vs_accuracy.json"))
    outputs["figures"] += [str(p) for p in plot_experiment_5_runtime_vs_D(
        res5, _fig_paths("fig5_runtime_vs_accuracy.pdf", "fig_step9_exp5_runtime_vs_D.png"))]
    outputs["figures"] += [str(p) for p in plot_experiment_5_runtime_vs_m(
        res5, _fig_paths("fig6_runtime_scaling_in_m.pdf", "fig_step9_exp5_runtime_vs_m.png"))]
    _done("experiment_5", t0)

    summary: dict[str, Any] = {
        "quick": bool(quick),
        "seed": int(seed),
        "use_cv": bool(use_cv),
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "hyperparameters": {
            k: {
                "gamma_rbf": b["gamma_rbf"],
                "gamma_laplacian": b["gamma_laplacian"],
                "lambda_svm": b["lambda_svm"],
            }
            for k, b in datasets.items()
        },
        "stage_seconds": stage_seconds,
        "total_seconds": float(sum(stage_seconds.values())),
        "outputs": outputs,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "blas_threads_env": os.environ.get("OPENBLAS_NUM_THREADS", ""),
        },
    }
    summary_path = res_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    outputs["results"].append(str(summary_path))
    return summary


def main(argv: list[str] | None = None) -> int:
    """
    Command-line interface of the full reproduction pipeline.

    Parameters
    ----------
    argv : list[str] or None, default=None
        Argument vector; None reads sys.argv[1:].

    Returns
    -------
    exit_code : int
        0 on success.
    """
    parser = argparse.ArgumentParser(description="Reproduce all experiments and figures of the SMML project.")
    parser.add_argument("--quick", action="store_true", help="Run the reduced smoke-test configuration.")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed (default: 42).")
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation and keep default hyperparameters.")
    parser.add_argument("--results-dir", type=str, default=None, help="JSON output directory.")
    parser.add_argument("--figures-dir", type=str, default=None, help="PDF figure output directory.")
    parser.add_argument("--vault-assets-dir", type=str, default="/home/davide/Work/SMML Project/vault/assets",
                        help="PNG output directory for Obsidian vault assets ('' disables PNG export).")
    args = parser.parse_args(argv)

    summary = run_pipeline(
        quick=args.quick,
        seed=args.seed,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        vault_assets_dir=args.vault_assets_dir or None,
        use_cv=not args.no_cv,
    )
    print(f"[run_all] finished in {summary['total_seconds']:.1f}s; "
          f"{len(summary['outputs']['results'])} result files, {len(summary['outputs']['figures'])} figure files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
