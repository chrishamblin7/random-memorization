#!/usr/bin/env python3
"""Train a random memorization model.

Usage:
    python -m experiments.train_random --config experiments/configs/powerlaw_N1k_transformer_b1.5.yaml
    python -m experiments.train_random --config experiments/configs/powerlaw_N1k_mlp_b1.5.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig
from src.train import train


def main():
    parser = argparse.ArgumentParser(description="Random memorization training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
