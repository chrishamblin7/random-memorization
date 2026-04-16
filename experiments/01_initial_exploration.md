# Experiment 1: Initial Exploration

## Goal

Test whether a small model can memorize random input-output pairs when
sampled with power-law frequency. Two architectures (1-layer transformer,
1-layer MLP) at two power-law exponents (beta=0.5, 1.5).

## Setup

- **n_examples**: 1,000
- **input_len**: 1,000, **output_len**: 1, **vocab**: 2 (binary)
- **Models**: 1-layer transformer (d_model=256, d_ff=1024) and 1-layer MLP (d_ff=1024)
- **Sampler**: power-law with beta in {0.5, 1.5}
- **Steps**: 200K, **batch_size**: 512

## Configs

- `powerlaw_N1k_transformer_b0.5.yaml`
- `powerlaw_N1k_transformer_b1.5.yaml`
- `powerlaw_N1k_mlp_b0.5.yaml`
- `powerlaw_N1k_mlp_b1.5.yaml`

## Results

- **MLP**: Learned quickly. With only 1K examples, both beta values reached
  near-perfect accuracy.
- **Transformer**: Failed to learn at input_len=1000. Accuracy stuck at ~50%
  (chance level). Root cause: with binary vocabulary and RoPE only on Q/K,
  the value path has only 2 distinct vectors across 1000 positions, making
  sequence identification impossible.

## Follow-up

- MLP retested at 100K examples (Experiment 2)
- Transformer debugged with shorter sequences (Experiment 3)

## Run directories

```
results/power_law_N1k_mlp_b0.5_s42/
results/power_law_N1k_mlp_b1.5_s42/
results/power_law_N1k_transformer_b0.5_s42/
results/power_law_N1k_transformer_b1.5_s42/
```
