#!/usr/bin/env python3
"""Generate scaling law plots from the L-shaped sweep.

Produces four plots:
  1. Kaplan-style L vs C  — one line per model size, showing loss decreasing then
     plateauing as a function of compute (params * steps).
  2. L vs N (convergence) — converged loss vs model params, log-log with power-law fit.
  3. L vs D (convergence) — converged loss vs dataset size, log-log with power-law fit.
  4. IsoFLOP curves       — L vs N for several fixed compute budgets.

Usage:
    python scripts/plot_scaling_laws.py --results-root results/
    python scripts/plot_scaling_laws.py --results-root results/ --out-dir scaling_plots/
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import curve_fit

mpl.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

LN2 = np.log(2.0)

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]


def load_run(results_dir):
    cfg_path = results_dir / "config.yaml"
    met_path = results_dir / "metrics.json"
    if not cfg_path.exists() or not met_path.exists():
        return None, None
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    try:
        with open(met_path) as f:
            metrics = json.load(f)
    except json.JSONDecodeError:
        raw = met_path.read_text()
        last_close = raw.rfind("}]")
        if last_close == -1:
            return None, None
        try:
            metrics = json.loads(raw[: last_close + 2])
        except json.JSONDecodeError:
            return None, None
    return cfg, metrics


def count_params(cfg):
    """Compute MLP parameter count from config."""
    input_len = cfg["input_len"]
    d_ff = cfg["d_ff"]
    output_vocab = cfg.get("output_vocab_size", 2)
    output_len = cfg.get("output_len", 1)
    return input_len * d_ff + d_ff + d_ff * (output_vocab * output_len) + (output_vocab * output_len)


def extract_trajectory(metrics):
    """Extract (steps, agg_loss_bits) arrays."""
    steps, losses = [], []
    for entry in metrics:
        if "agg_loss" in entry:
            steps.append(entry["step"])
            losses.append(entry["agg_loss"] / LN2)
    return np.array(steps), np.array(losses)


def converged_loss(losses, frac=0.2):
    """Min of the last `frac` of eval losses — robust to noise."""
    tail = max(1, int(len(losses) * frac))
    return np.min(losses[-tail:])


def _ema(values, alpha=0.05):
    result = np.empty(len(values))
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]
    return result


def _power_law(x, a, alpha, c):
    return a * x ** (-alpha) + c


def _fit_power_law(x, y):
    """Fit y = a * x^{-alpha} + c. Returns (a, alpha, c) or None."""
    try:
        popt, _ = curve_fit(
            _power_law, x, y,
            p0=[1.0, 0.5, 0.0],
            bounds=([0, 0, 0], [np.inf, 5.0, np.inf]),
            maxfev=10000,
        )
        return popt
    except (RuntimeError, ValueError):
        return None


def discover_scaling_runs(results_root, filter_input_len=128):
    """Find runs matching the L-shaped sweep naming convention.

    Only includes runs with the specified input_len to avoid picking up
    results from other sweeps that share the same (n_examples, d_ff).
    """
    runs = {}
    pattern = re.compile(r"mlp_N(\d+)k_L(\d+)_D(\d+)_b([\d.]+)_s(\d+)")
    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        n_ex = int(m.group(1)) * 1000
        il = int(m.group(2))
        dff = int(m.group(3))
        beta = float(m.group(4))
        if filter_input_len is not None and il != filter_input_len:
            continue
        cfg, metrics = load_run(d)
        if cfg is None:
            continue
        steps, losses = extract_trajectory(metrics)
        if len(steps) < 5:
            continue
        n_params = count_params(cfg)
        runs[(n_ex, dff)] = {
            "dir": d, "cfg": cfg, "metrics": metrics,
            "steps": steps, "losses": losses, "n_params": n_params,
            "n_examples": n_ex, "d_ff": dff, "beta": beta,
            "input_len": il,
        }
    return runs


def plot_loss_vs_compute(runs, out_dir, fixed_n_examples=50000):
    """Plot 1: Kaplan-style L vs C — one line per model size."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    arm1 = {k: v for k, v in runs.items() if v["n_examples"] == fixed_n_examples}
    arm1_sorted = sorted(arm1.items(), key=lambda kv: kv[1]["d_ff"])

    for i, ((_, _), run) in enumerate(arm1_sorted):
        color = COLORS[i % len(COLORS)]
        compute = run["steps"] * run["n_params"]
        smoothed = _ema(run["losses"], alpha=0.05)
        ax.plot(compute, smoothed, color=color, linewidth=1.8,
                label=f'd_ff={run["d_ff"]} ({run["n_params"]:,}p)')
        ax.plot(compute, run["losses"], color=color, alpha=0.15, linewidth=0.5)

    envelope_x, envelope_y = [], []
    for (_, _), run in arm1_sorted:
        compute = run["steps"] * run["n_params"]
        for c, l in zip(compute, run["losses"]):
            envelope_x.append(c)
            envelope_y.append(l)
    if envelope_x:
        order = np.argsort(envelope_x)
        ex = np.array(envelope_x)[order]
        ey = np.array(envelope_y)[order]
        running_min = np.minimum.accumulate(ey)
        ax.plot(ex, running_min, "k--", linewidth=1.5, alpha=0.6, label="Compute frontier")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute  (params × steps)")
    ax.set_ylabel("Loss (bits)")
    ax.set_title(f"Loss vs Compute  (N_examples={fixed_n_examples:,}, β=1.5)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_vs_compute.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'loss_vs_compute.png'}")


def plot_loss_vs_model_size(runs, out_dir, fixed_n_examples=50000):
    """Plot 2: converged loss vs model params, log-log."""
    fig, ax = plt.subplots(figsize=(7, 5))

    arm1 = {k: v for k, v in runs.items() if v["n_examples"] == fixed_n_examples}
    points = sorted(arm1.values(), key=lambda r: r["n_params"])
    if not points:
        return

    xs = np.array([r["n_params"] for r in points])
    ys = np.array([converged_loss(r["losses"]) for r in points])

    ax.scatter(xs, ys, s=60, zorder=5, color=COLORS[0], edgecolors="black", linewidths=0.5)

    fit = _fit_power_law(xs, ys)
    if fit is not None:
        a, alpha, c = fit
        x_fit = np.geomspace(xs.min() * 0.8, xs.max() * 1.2, 200)
        y_fit = _power_law(x_fit, a, alpha, c)
        ax.plot(x_fit, y_fit, "--", color=COLORS[1], linewidth=1.5,
                label=f"Fit: L = {a:.2f}·N^{{-{alpha:.2f}}} + {c:.3f}")
        x_theory = np.geomspace(xs.min() * 0.8, xs.max() * 1.2, 200)
        y_theory = _power_law(x_theory, a, 0.5, c)
        ax.plot(x_theory, y_theory, ":", color="gray", linewidth=1.2, alpha=0.7,
                label="Theory (α=0.5)")

    for r in points:
        ax.annotate(f'd={r["d_ff"]}', (r["n_params"], converged_loss(r["losses"])),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model Parameters (N)")
    ax.set_ylabel("Converged Loss (bits)")
    ax.set_title(f"Loss vs Model Size  (N_examples={fixed_n_examples:,}, β=1.5)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_vs_model_size.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'loss_vs_model_size.png'}")


def plot_loss_vs_dataset_size(runs, out_dir, fixed_d_ff=512):
    """Plot 3: converged loss vs dataset size, log-log."""
    fig, ax = plt.subplots(figsize=(7, 5))

    arm2 = {k: v for k, v in runs.items() if v["d_ff"] == fixed_d_ff}
    points = sorted(arm2.values(), key=lambda r: r["n_examples"])
    if not points:
        return

    xs = np.array([r["n_examples"] for r in points], dtype=float)
    ys = np.array([converged_loss(r["losses"]) for r in points])

    ax.scatter(xs, ys, s=60, zorder=5, color=COLORS[2], edgecolors="black", linewidths=0.5)

    if len(xs) >= 3:
        fit = _fit_power_law(1.0 / xs, ys)
        if fit is not None:
            a, alpha, c = fit
            x_fit = np.geomspace(xs.min() * 0.8, xs.max() * 1.2, 200)
            y_fit = _power_law(1.0 / x_fit, a, alpha, c)
            ax.plot(x_fit, y_fit, "--", color=COLORS[3], linewidth=1.5,
                    label=f"Fit: L = {a:.2f}·D^{{{alpha:.2f}}} + {c:.3f}")

    for r in points:
        n_str = f'{r["n_examples"]//1000}k' if r["n_examples"] >= 1000 else str(r["n_examples"])
        ax.annotate(f'N={n_str}', (r["n_examples"], converged_loss(r["losses"])),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset Size (D)")
    ax.set_ylabel("Converged Loss (bits)")
    ax.set_title(f"Loss vs Dataset Size  (d_ff={fixed_d_ff}, β=1.5)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_vs_dataset_size.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'loss_vs_dataset_size.png'}")


def plot_isoflop(runs, out_dir, fixed_n_examples=50000, n_budgets=5):
    """Plot 4: IsoFLOP curves — L vs N for several fixed compute budgets."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    arm1 = {k: v for k, v in runs.items() if v["n_examples"] == fixed_n_examples}
    if not arm1:
        return

    all_compute = []
    for run in arm1.values():
        c_max = run["steps"][-1] * run["n_params"]
        all_compute.append(c_max)
    c_min_global = min(r["steps"][1] * r["n_params"] for r in arm1.values() if len(r["steps"]) > 1)
    c_max_global = min(all_compute)

    budgets = np.geomspace(c_min_global * 2, c_max_global * 0.8, n_budgets)

    arm1_sorted = sorted(arm1.values(), key=lambda r: r["n_params"])

    for bi, budget in enumerate(budgets):
        color = COLORS[bi % len(COLORS)]
        ns, ls = [], []
        for run in arm1_sorted:
            compute = run["steps"] * run["n_params"]
            valid = compute <= budget
            if not valid.any():
                continue
            loss_at_budget = run["losses"][valid][-1]
            ns.append(run["n_params"])
            ls.append(loss_at_budget)
        if len(ns) >= 2:
            budget_str = f"{budget:.1e}"
            ax.plot(ns, ls, "o-", color=color, markersize=5, linewidth=1.5,
                    label=f"C = {budget_str}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model Parameters (N)")
    ax.set_ylabel("Loss (bits)")
    ax.set_title(f"IsoFLOP Curves  (N_examples={fixed_n_examples:,}, β=1.5)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "isoflop_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'isoflop_curves.png'}")


def main():
    parser = argparse.ArgumentParser(description="Plot scaling laws from L-shaped sweep")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--out-dir", type=str, default="scaling_plots")
    parser.add_argument("--fixed-n-examples", type=int, default=50000)
    parser.add_argument("--fixed-d-ff", type=int, default=512)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_scaling_runs(results_root)
    if not runs:
        print("No completed scaling runs found.")
        return

    print(f"Found {len(runs)} runs:")
    for (n_ex, dff), run in sorted(runs.items()):
        cl = converged_loss(run["losses"])
        print(f"  N={n_ex:>7,}  d_ff={dff:>5}  params={run['n_params']:>8,}  "
              f"converged_loss={cl:.4f} bits  steps={run['steps'][-1]}")

    plot_loss_vs_compute(runs, out_dir, args.fixed_n_examples)
    plot_loss_vs_model_size(runs, out_dir, args.fixed_n_examples)
    plot_loss_vs_dataset_size(runs, out_dir, args.fixed_d_ff)
    plot_isoflop(runs, out_dir, args.fixed_n_examples)

    print(f"\nDone. 4 scaling law plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
