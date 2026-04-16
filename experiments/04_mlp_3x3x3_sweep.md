# Experiment 4: MLP 3x3x3 Factorial Sweep

## Goal

Systematically explore the interaction between input length, power-law
exponent, and model width on memorization dynamics. Find a regime with
non-trivial residual loss suitable for scaling law analysis.

## Setup

- **n_examples**: 50,000
- **Factorial axes**:
  - `input_len` in {128, 256, 512}
  - `beta` in {0.5, 1.0, 1.5}
  - `d_ff` in {256, 512, 1024}
- **Model**: 1-layer MLP
- **Steps**: 100K, **batch_size**: 2048
- **Total**: 27 experiments on 4 GPUs

## Configs

`mlp_N50k_L{128,256,512}_D{256,512,1024}_b{0.5,1.0,1.5}.yaml` (27 files)

## Key Results

**beta=0.5**: All configs hit 100% accuracy and ~0 loss. Sampling is too
uniform; every example gets enough exposure.

**beta=1.0**: Nearly saturated. Only the largest configs show tiny residual
loss (~0.004).

**beta=1.5**: The productive regime. Final loss at 100K steps:

|          | D256  | D512  | D1024 |
| -------- | ----- | ----- | ----- |
| **L128** | 0.560 | 0.293 | 0.181 |
| **L256** | 0.219 | 0.158 | 0.143 |
| **L512** | 0.147 | 0.118 | 0.111 |

Loss clearly decreases with d_ff (model size). input_len=128 gives the
widest spread in loss across model sizes (3x range), making it the best
regime for measuring scaling exponents.

## Plots

Sweep grid plots in `plots/` directory — 3x3 grids holding each axis
constant, in linear, log-log, and log-x scale variants.

## Run directories

```
results/mlp_N50k_L{128,256,512}_D{256,512,1024}_b{0.5,1.0,1.5}_s42/
```

## Command

```bash
cd ~/projects/random-memorization
bash scripts/run_sweep.sh experiments/configs/mlp_N50k_*.yaml
```
