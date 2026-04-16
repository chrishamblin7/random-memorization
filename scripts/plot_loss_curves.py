#!/usr/bin/env python3
"""Plot per-example loss curves for random memorization experiments.

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


def load_run(results_dir: Path):
    with open(results_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(results_dir / "metrics.json") as f:
        metrics = json.load(f)
    return cfg, metrics


def extract_curves(metrics, example_ids, prob_of=None):
    steps = [e["step"] for e in metrics if "per_example" in e]
    losses = {i: [] for i in example_ids}
    weighted_loss = []

    if prob_of is not None:
        weights = np.array([prob_of[i] for i in example_ids])
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(example_ids)) / len(example_ids)

    for entry in metrics:
        if "per_example" not in entry:
            continue
        pe = entry["per_example"]
        vals = []
        for i in example_ids:
            loss_bits = pe[str(i)]["loss"] / LN2
            losses[i].append(loss_bits)
            vals.append(loss_bits)
        weighted_loss.append(np.dot(weights, vals))

    return np.array(steps), losses, np.array(weighted_loss)


def build_color_mapping(example_ids, results_dir, cfg):
    probs_path = results_dir / "sampler_probs.json"
    if probs_path.exists():
        with open(probs_path) as f:
            raw = json.load(f)
        prob_of = {int(k): v for k, v in raw.items()}
    else:
        beta = cfg.get("beta", 1.5)
        idx = np.array(example_ids, dtype=np.float64)
        w = (idx + 1.0) ** (-beta)
        w /= w.sum()
        prob_of = dict(zip(example_ids, w.tolist()))

    probs = np.array([prob_of[i] for i in example_ids])
    cmap = mpl.colormaps["viridis"].reversed()
    norm = mpl.colors.LogNorm(vmin=probs.min(), vmax=probs.max())
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    return sm, prob_of


def _add_colorbar(fig, ax, sm, example_ids, prob_of):
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Example index (colored by sampling probability)")
    sorted_by_prob = sorted(example_ids, key=lambda i: prob_of[i])
    tick_ids = sorted_by_prob[:: max(1, len(sorted_by_prob) // 8)]
    if sorted_by_prob[-1] not in tick_ids:
        tick_ids.append(sorted_by_prob[-1])
    cbar.set_ticks([prob_of[i] for i in tick_ids])
    cbar.set_ticklabels([str(i) for i in tick_ids])
    return cbar


def _ema(values, alpha=0.05):
    result = np.empty(len(values))
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]
    return result


def make_static_plot(
    steps, losses, mean_loss, example_ids, sm, prob_of, results_dir,
    log_y=False, prefix="loss_curves", ylabel="Loss (bits)",
    ema_alpha=0.05,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    for i in example_ids:
        color = sm.to_rgba(prob_of[i])
        smoothed = _ema(np.array(losses[i]), alpha=ema_alpha)
        ax.plot(steps, smoothed, color=color, alpha=0.4, linewidth=0.5)

    ax.plot(steps, mean_loss, color="red", linewidth=2.5, label="Weighted mean loss")

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")

    _add_colorbar(fig, ax, sm, example_ids, prob_of)

    fig.tight_layout()
    suffix = "_logy" if log_y else ""
    out = results_dir / f"{prefix}{suffix}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot per-example loss curves")
    parser.add_argument("--results-dir", type=str, required=True)
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

    example_ids = sorted(int(k) for k in first_pe.keys())
    sm, prob_of = build_color_mapping(example_ids, results_dir, cfg)
    steps, losses, mean_loss = extract_curves(metrics, example_ids, prob_of)

    make_static_plot(steps, losses, mean_loss, example_ids, sm, prob_of, results_dir,
                     log_y=False, prefix="loss_curves", ylabel="Loss (bits)")
    make_static_plot(steps, losses, mean_loss, example_ids, sm, prob_of, results_dir,
                     log_y=True, prefix="loss_curves", ylabel="Loss (bits)")


if __name__ == "__main__":
    main()
