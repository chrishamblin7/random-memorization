import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RoPE helpers (from successor-quanta)
# ---------------------------------------------------------------------------

def _build_rope_cache(seq_len: int, head_dim: int, device: torch.device):
    theta = 10000.0
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


# ---------------------------------------------------------------------------
# Transformer block (from successor-quanta, bidirectional)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                rope_cos: torch.Tensor | None = None,
                rope_sin: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)

        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None:
            q = _apply_rope(q, rope_cos, rope_sin)
            k = _apply_rope(k, rope_cos, rope_sin)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.o_proj(attn_out)

        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# MemorizationTransformer
# ---------------------------------------------------------------------------

class MemorizationTransformer(nn.Module):
    """Bidirectional transformer encoder for memorization.

    Reads input_len tokens, outputs logits for the last output_len positions.
    Uses learned positional embeddings PLUS RoPE so that position information
    flows into values (not just Q/K). Without this, binary-vocab inputs produce
    only 2 distinct value vectors across all positions, making sequence
    identification impossible.
    """

    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        input_len: int,
        output_len: int = 1,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 1,
        d_ff: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.tok_emb = nn.Embedding(input_vocab_size, d_model)
        self.pos_emb = nn.Embedding(input_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)

        rope_cos, rope_sin = _build_rope_cache(T, self.head_dim, x.device)
        for block in self.blocks:
            h = block(h, rope_cos, rope_sin)

        h = self.ln_f(h)
        return self.head(h[:, -self.output_len:])


# ---------------------------------------------------------------------------
# MemorizationMLP (Michaud et al. 2023 style)
# ---------------------------------------------------------------------------

class MemorizationMLP(nn.Module):
    """Single-hidden-layer ReLU MLP for memorization.

    Flattens input tokens to a float vector, passes through one hidden layer.
    Following Michaud et al. (2023) Section 3.2 multitask sparse parity setup.
    """

    def __init__(
        self,
        input_len: int,
        output_vocab_size: int,
        output_len: int = 1,
        d_ff: int = 1024,
        **kwargs,
    ):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.output_vocab_size = output_vocab_size

        self.net = nn.Sequential(
            nn.Linear(input_len, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, output_vocab_size * output_len),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = x.float()
        out = self.net(h)
        return out.view(B, self.output_len, self.output_vocab_size)


def build_model(cfg, device: str = "cpu") -> nn.Module:
    if cfg.model_type == "transformer":
        return MemorizationTransformer(
            input_vocab_size=cfg.input_vocab_size,
            output_vocab_size=cfg.output_vocab_size,
            input_len=cfg.input_len,
            output_len=cfg.output_len,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
        ).to(device)
    elif cfg.model_type == "mlp":
        return MemorizationMLP(
            input_len=cfg.input_len,
            output_vocab_size=cfg.output_vocab_size,
            output_len=cfg.output_len,
            d_ff=cfg.d_ff,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")
