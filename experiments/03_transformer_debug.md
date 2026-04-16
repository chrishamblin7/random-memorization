# Experiment 3: Transformer Debug (Short Sequences)

## Goal

Debug why the 1-layer transformer failed to learn at input_len=1000.
Test with shorter sequences to isolate the issue.

## Setup

- **n_examples**: 1,000
- **input_len**: 50 and 100, **output_len**: 1, **vocab**: 2
- **Model**: 1-layer transformer (d_model=256, n_heads=4, d_ff=1024)
- **Sampler**: power-law, beta=1.5
- **Steps**: 200K, **batch_size**: 512

## Configs

- `debug_transformer_len50.yaml`
- `debug_transformer_len100.yaml`

## Diagnosis

Even at shorter sequences, the transformer struggled. The root cause was
identified: with binary vocabulary (vocab_size=2) and RoPE applied only to
Q and K, the value vectors `W_v @ emb_0` and `W_v @ emb_1` are the same
regardless of position. The attention output becomes a weighted sum of only
2 distinct vectors, which is insufficient to distinguish among sequences.

## Fix

Added learned positional embeddings (`nn.Embedding(input_len, d_model)`)
summed with token embeddings in the forward pass. This ensures positional
information flows into the value path, not just Q/K.

## Run directories

```
results/  (debug runs, short-lived)
```
