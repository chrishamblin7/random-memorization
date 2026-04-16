#!/usr/bin/env python3
"""Plot binned loss curves for random memorization experiments.

When n_examples is large, individual per-example lines are unreadable.
Instead, group examples into log-spaced bins (finer bins for the most
frequently sampled examples) and plot the mean loss of each bin.

Usage:
    python scripts/plot_loss_curves.py --results-dir results/<run_name>
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

LN2 = np.log(2.0)
TARGET_BINS = 200


def load_run(results_dir: Path):
    with open(results_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(results_dir / "metrics.json") as f:
        metrics = json.load(f)
    return cfg, metrics


def _make_bin_edges(n_examples: int, target_bins: int = TARGET_BINS):
    """Log-spaced bin edges: small bins for low-index (frequent) examples."""
    edges = np.unique(np.geomspace(1, n_examples, num=target_bins + 1).astype(int))
    edges[0] = 0
    edges[-1] = n_examples
    return edges


def extract_binned_curves(metrics, bin_edges, prob_of, n_examples):
    """Extract per-bin mean loss curves and a weighted-mean curve."""
    steps = [e["step"] for e in metrics if "per_example" in e]
    n_bins = len(bin_edges) - 1

    all_probs = np.array([prob_of.get(i, 0.0) for i in range(n_examples)])
    total_prob = all_probs.sum()

    bin_mean_probs = np.zeros(n_bins)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        bin_mean_probs[b] = all_probs[lo:hi].mean()

    bin_losses = [[] for _ in range(n_bins)]
    weighted_loss = []

    for entry in metrics:
        if "per_example" not in entry:
            continue
        pe = entry["per_example"]

        all_losses = np.zeros(n_examples)
        for i in range(n_examples):
            all_losses[i] = pe[str(i)]["loss"] / LN2

        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            bin_losses[b].append(all_losses[lo:hi].mean())

        weighted_loss.append(np.dot(all_probs / total_prob, all_losses))

    return np.array(steps), bin_losses, np.array(weighted_loss), bin_mean_probs


def build_bin_color_mapping(bin_mean_probs):
    cmap = mpl.colormaps["viridis"].reversed()
    norm = mpl.colors.LogNorm(
        vmin=bin_mean_probs[bin_mean_probs > 0].min(),
        vmax=bin_mean_probs.max(),
    )
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm


def _add_colorbar(fig, ax, sm, bin_edges, bin_mean_probs):
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Example rank (colored by sampling probability)")
    n_ticks = 8
    idx = np.linspace(0, len(bin_mean_probs) - 1, n_ticks, dtype=int)
    tick_probs = bin_mean_probs[idx]
    tick_labels = []
    for i in idx:
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi - lo == 1:
            tick_labels.append(str(lo))
        else:
            tick_labels.append(f"{lo}-{hi-1}")
    valid = tick_probs > 0
    cbar.set_ticks(tick_probs[valid])
    cbar.set_ticklabels([l for l, v in zip(tick_labels, valid) if v])
    return cbar


def _ema(values, alpha=0.05):
    result = np.empty(len(values))
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]
    return result


def make_static_plot(
    steps, bin_losses, mean_loss, bin_edges, bin_mean_probs, sm, results_dir,
    log_y=False, prefix="loss_curves", ylabel="Loss (bits)", ema_alpha=0.05,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    n_bins = len(bin_losses)
    for b in range(n_bins):
        p = bin_mean_probs[b]
        if p <= 0:
            continue
        color = sm.to_rgba(p)
        smoothed = _ema(np.array(bin_losses[b]), alpha=ema_alpha)
        ax.plot(steps, smoothed, color=color, alpha=0.4, linewidth=0.5)

    ax.plot(steps, mean_loss, color="red", linewidth=2.5, label="Weighted mean loss")

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")

    _add_colorbar(fig, ax, sm, bin_edges, bin_mean_probs)

    fig.tight_layout()
    suffix = "_logy" if log_y else ""
    out = results_dir / f"{prefix}{suffix}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot binned loss curves")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--bins", type=int, default=TARGET_BINS)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cfg, metrics = load_run(results_dir)

    first_pe = None
    for entry in metrics:
        if "per_example" in entry:
            first_pe = entry["per_example"]
            break
    if first_pe is None:
        print("No per_example data found in metrics.")
        return

    n_examples = len(first_pe)
    example_ids = sorted(int(k) for k in first_pe.keys())

    probs_path = results_dir / "sampler_probs.json"
    if probs_path.exists():
        with open(probs_path) as f:
            raw = json.load(f)
        prob_of = {int(k): v for k, v in raw.items()}
    else:
        beta = cfg.get("beta", 1.5)
        idx = np.arange(n_examples, dtype=np.float64)
        w = (idx + 1.0) ** (-beta)
        w /= w.sum()
        prob_of = {i: float(p) for i, p in enumerate(w)}

    if n_examples <= args.bins * 2:
        bin_edges = np.arange(n_examples + 1)
    else:
        bin_edges = _make_bin_edges(n_examples, target_bins=args.bins)

    steps, bin_losses, mean_loss, bin_mean_probs = extract_binned_curves(
        metrics, bin_edges, prob_of, n_examples
    )

    sm = build_bin_color_mapping(bin_mean_probs)

    make_static_plot(steps, bin_losses, mean_loss, bin_edges, bin_mean_probs, sm,
                     results_dir, log_y=False)
    make_static_plot(steps, bin_losses, mean_loss, bin_edges, bin_mean_probs, sm,
                     results_dir, log_y=True)


if __name__ == "__main__":
    main()
