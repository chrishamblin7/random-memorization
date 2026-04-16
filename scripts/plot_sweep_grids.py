#!/usr/bin/env python3
"""Generate 3x3 grid plots for the factorial MLP sweep.

For each of the 3 sweep axes (input_len, beta, d_ff), hold that axis constant
and plot the other two as a 3x3 subplot grid. Each subplot shows the binned
loss curves for that parameter combination.

Produces 3 scale variants per grid:
  - linear:  linear x, linear y
  - loglog:  log x, log y
  - logx:    log x, linear y

Usage:
    python scripts/plot_sweep_grids.py --results-root results/
    python scripts/plot_sweep_grids.py --results-root results/ --out-dir plots/
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

LN2 = np.log(2.0)
TARGET_BINS = 200


def _make_bin_edges(n_examples, target_bins=TARGET_BINS):
    edges = np.unique(np.geomspace(1, n_examples, num=target_bins + 1).astype(int))
    edges[0] = 0
    edges[-1] = n_examples
    return edges


def _ema(values, alpha=0.05):
    result = np.empty(len(values))
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]
    return result


def load_run(results_dir):
    cfg_path = results_dir / "config.yaml"
    met_path = results_dir / "metrics.json"
    if not cfg_path.exists() or not met_path.exists():
        return None, None
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    with open(met_path) as f:
        metrics = json.load(f)
    return cfg, metrics


def extract_weighted_mean(metrics, n_examples, prob_of):
    all_probs = np.array([prob_of.get(i, 0.0) for i in range(n_examples)])
    total_prob = all_probs.sum()
    if total_prob == 0:
        total_prob = 1.0

    steps = []
    weighted = []
    for entry in metrics:
        if "per_example" not in entry:
            continue
        pe = entry["per_example"]
        steps.append(entry["step"])
        losses = np.array([pe[str(i)]["loss"] / LN2 for i in range(n_examples)])
        weighted.append(np.dot(all_probs / total_prob, losses))
    return np.array(steps), np.array(weighted)


def extract_binned(metrics, n_examples, prob_of, target_bins=TARGET_BINS):
    all_probs = np.array([prob_of.get(i, 0.0) for i in range(n_examples)])
    total_prob = max(all_probs.sum(), 1e-30)

    if n_examples <= target_bins * 2:
        bin_edges = np.arange(n_examples + 1)
    else:
        bin_edges = _make_bin_edges(n_examples, target_bins)

    n_bins = len(bin_edges) - 1
    bin_mean_probs = np.zeros(n_bins)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        bin_mean_probs[b] = all_probs[lo:hi].mean()

    steps = []
    bin_losses = [[] for _ in range(n_bins)]
    weighted = []

    for entry in metrics:
        if "per_example" not in entry:
            continue
        pe = entry["per_example"]
        steps.append(entry["step"])
        all_l = np.array([pe[str(i)]["loss"] / LN2 for i in range(n_examples)])
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            bin_losses[b].append(all_l[lo:hi].mean())
        weighted.append(np.dot(all_probs / total_prob, all_l))

    return np.array(steps), bin_losses, np.array(weighted), bin_edges, bin_mean_probs


def get_prob_of(results_dir, cfg, n_examples):
    probs_path = results_dir / "sampler_probs.json"
    if probs_path.exists():
        with open(probs_path) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    beta = cfg.get("beta", 1.5)
    idx = np.arange(n_examples, dtype=np.float64)
    w = (idx + 1.0) ** (-beta)
    w /= w.sum()
    return {i: float(p) for i, p in enumerate(w)}


def discover_runs(results_root):
    """Find all mlp_N50k runs and parse their parameters."""
    runs = {}
    pattern = re.compile(r"mlp_N\d+k_L(\d+)_D(\d+)_b([\d.]+)_s\d+")
    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        il, dff, beta = int(m.group(1)), int(m.group(2)), float(m.group(3))
        cfg, metrics = load_run(d)
        if cfg is None or not any("per_example" in e for e in metrics):
            continue
        runs[(il, beta, dff)] = (d, cfg, metrics)
    return runs


def plot_grid(runs, held_axis, held_value, row_values, col_values,
              row_label, col_label, key_fn, out_dir, scale="linear"):
    """Plot a 3x3 grid of subplots.

    held_axis: name of the held-constant axis (for the title)
    row_values / col_values: lists of 3 values for rows / columns
    key_fn(row_val, col_val) -> (input_len, beta, d_ff) tuple
    """
    fig, axes = plt.subplots(len(row_values), len(col_values),
                             figsize=(5 * len(col_values), 4 * len(row_values)),
                             squeeze=False)

    for ri, rv in enumerate(row_values):
        for ci, cv in enumerate(col_values):
            ax = axes[ri][ci]
            key = key_fn(rv, cv)
            if key not in runs:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
                ax.set_title(f"{row_label}={rv}, {col_label}={cv}", fontsize=9)
                continue

            rdir, cfg, metrics = runs[key]
            n_examples = cfg.get("n_examples", 50000)
            prob_of = get_prob_of(rdir, cfg, n_examples)

            steps, bin_losses, weighted, bin_edges, bin_mean_probs = extract_binned(
                metrics, n_examples, prob_of
            )

            if len(steps) == 0:
                ax.text(0.5, 0.5, "No eval data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
                ax.set_title(f"{row_label}={rv}, {col_label}={cv}", fontsize=9)
                continue

            cmap = mpl.colormaps["viridis"].reversed()
            valid_probs = bin_mean_probs[bin_mean_probs > 0]
            if len(valid_probs) == 0:
                ax.set_title(f"{row_label}={rv}, {col_label}={cv}", fontsize=9)
                continue
            norm = mpl.colors.LogNorm(vmin=valid_probs.min(), vmax=valid_probs.max())

            for b in range(len(bin_losses)):
                p = bin_mean_probs[b]
                if p <= 0:
                    continue
                color = cmap(norm(p))
                smoothed = _ema(np.array(bin_losses[b]), alpha=0.05)
                ax.plot(steps, smoothed, color=color, alpha=0.35, linewidth=0.4)

            ax.plot(steps, weighted, color="red", linewidth=1.5)

            if scale == "loglog":
                ax.set_xscale("log")
                ax.set_yscale("log")
            elif scale == "logx":
                ax.set_xscale("log")

            ax.set_title(f"{row_label}={rv}, {col_label}={cv}", fontsize=9)
            if ri == len(row_values) - 1:
                ax.set_xlabel("Steps", fontsize=8)
            if ci == 0:
                ax.set_ylabel("Loss (bits)", fontsize=8)
            ax.tick_params(labelsize=7)

    scale_label = {"linear": "linear", "loglog": "log-log", "logx": "log-x"}[scale]
    fig.suptitle(f"{held_axis} = {held_value}  ({scale_label})", fontsize=13, y=1.01)
    fig.tight_layout()

    safe_held = str(held_value).replace(".", "p")
    fname = f"grid_{held_axis}_{safe_held}_{scale}.png"
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot 3x3 sweep grids")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--out-dir", type=str, default="plots")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(results_root)
    if not runs:
        print("No completed sweep runs found.")
        return

    input_lens = sorted({k[0] for k in runs})
    betas = sorted({k[1] for k in runs})
    d_ffs = sorted({k[2] for k in runs})

    print(f"Found {len(runs)} runs: input_len={input_lens}, beta={betas}, d_ff={d_ffs}")

    scales = ["linear", "loglog", "logx"]

    for il in input_lens:
        for sc in scales:
            plot_grid(runs, held_axis="input_len", held_value=il,
                      row_values=betas, col_values=d_ffs,
                      row_label="beta", col_label="d_ff",
                      key_fn=lambda rv, cv, _il=il: (_il, rv, cv),
                      out_dir=out_dir, scale=sc)

    for beta in betas:
        for sc in scales:
            plot_grid(runs, held_axis="beta", held_value=beta,
                      row_values=input_lens, col_values=d_ffs,
                      row_label="input_len", col_label="d_ff",
                      key_fn=lambda rv, cv, _b=beta: (rv, _b, cv),
                      out_dir=out_dir, scale=sc)

    for dff in d_ffs:
        for sc in scales:
            plot_grid(runs, held_axis="d_ff", held_value=dff,
                      row_values=input_lens, col_values=betas,
                      row_label="input_len", col_label="beta",
                      key_fn=lambda rv, cv, _d=dff: (rv, cv, _d),
                      out_dir=out_dir, scale=sc)

    print(f"\nDone. {len(input_lens)*3 + len(betas)*3 + len(d_ffs)*3} grid plots in {out_dir}/")


if __name__ == "__main__":
    main()
