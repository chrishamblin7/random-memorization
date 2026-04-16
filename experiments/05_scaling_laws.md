# Experiment 5: Memorization Scaling Laws (L-Shaped Sweep)

## Goal

Establish power-law scaling of loss with model size and dataset size for
random memorization under power-law sampling. Produce Kaplan-style
compute-optimal plots.

## Theoretical Prediction

For power-law sampling with exponent beta, example `i` gets probability
`P(i) = (i+1)^{-beta} / Z`. If the model memorizes the top-k most frequent
examples, residual loss scales as `L(k) ~ k^{1-beta}`. For beta=1.5 and
capacity k proportional to model parameters N:

```
L(N) ~ N^{-0.5}
```

## Design: L-Shaped Sweep

Fixed: beta=1.5, input_len=128, output_len=1, vocab=2, lr=0.001

**Arm 1 -- Model size sweep** (n_examples=50K):
  d_ff in {64, 128, 256, 512, 1024, 2048} -- 6 experiments

**Arm 2 -- Dataset size sweep** (d_ff=512):
  n_examples in {2K, 10K, 50K, 200K} -- 3 new experiments (50K shared)

**Total: 9 experiments.** d_ff=512 / n_examples=50K is the intersection.

## Training Protocol

- **Steps**: 200K with cosine LR schedule (1K warmup, peak lr=0.001)
- **Batch size**: 2048
- **Eval every**: 500 steps (aggregate only; per-example at final step)
- **Lite metrics**: per_example logged only at step 200K to keep files small
- **Instance**: chris-sq-4x (4x A100-40GB)

## Configs

```
experiments/configs/scaling_N50k_D{64,128,256,512,1024,2048}.yaml
experiments/configs/scaling_N{2k,10k,200k}_D512.yaml
```

## Results

### Arm 1: Loss vs Model Size (n_examples=50K)

| d_ff | Params   | Loss    | Acc     |
|------|----------|---------|---------|
| 64   | 8,386    | 1.0067  | 64.25%  |
| 128  | 16,770   | 0.7559  | 73.26%  |
| 256  | 33,538   | 0.3106  | 90.08%  |
| 512  | 67,074   | 0.0199  | 99.71%  |
| 1024 | 134,146  | 0.0022  | 99.97%  |
| 2048 | 268,290  | 0.0003  | 100.0%  |

**Fitted power law**: L = 5787 * N^{-0.91} + 0.000
(theoretical prediction: exponent = -0.5)

### Arm 2: Loss vs Dataset Size (d_ff=512)

| n_examples | Loss    | Acc     |
|------------|---------|---------|
| 2,000      | ~0      | 100.0%  |
| 10,000     | ~0      | 100.0%  |
| 50,000     | 0.0199  | 99.71%  |
| 200,000    | 0.8185  | 68.83%  |

## Analysis

The model-size scaling exponent (-0.91) is steeper than the theoretical
prediction (-0.5). This suggests either:
1. Model capacity (number of memorizable examples) scales superlinearly
   with d_ff for a 1-layer MLP, or
2. The assumption that examples are memorized strictly in frequency order
   is too simplistic -- the model may partially learn multiple examples
   simultaneously.

The dataset-size arm shows the expected behavior: at small D (2K, 10K),
the model fully memorizes everything; at large D (200K), significant
residual loss remains.

## Plots

Scaling law plots in `scaling_plots/`:
- `loss_vs_compute.png` -- Kaplan-style L vs C (one line per model size)
- `loss_vs_model_size.png` -- converged L vs N, log-log with power-law fit
- `loss_vs_dataset_size.png` -- converged L vs D, log-log
- `isoflop_curves.png` -- IsoFLOP analysis

## Command

```bash
cd ~/projects/random-memorization
bash scripts/run_sweep.sh experiments/configs/scaling_*.yaml
```
