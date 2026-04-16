from dataclasses import dataclass, asdict
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    # --- Data ---
    n_examples: int = 1000
    input_len: int = 1000
    output_len: int = 1
    input_vocab_size: int = 2
    output_vocab_size: int = 2
    data_seed: int = 0

    # --- Sampling ---
    beta: float = 1.5
    sampler_type: str = "power_law"  # "power_law" or "uniform"

    # --- Model ---
    model_type: str = "transformer"  # "transformer" or "mlp"
    n_layers: int = 1
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0
    pos_emb_type: str = "rope"

    # --- Training ---
    lr: float = 0.001
    weight_decay: float = 0.0
    loss_type: str = "cross_entropy"  # "cross_entropy" or "stablemax"
    num_steps: int = 200000
    batch_size: int = 512
    eval_every: int = 1000
    checkpoint_every: int = 10000

    # --- Infrastructure ---
    seed: int = 42
    run_name: str = ""
    gcs_bucket: str = "/cloud/misc/chris/random-memorization/"
    device: str = "cuda"
    wandb_project: str = "random-memorization"
    resume: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def auto_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        n_str = f"{self.n_examples // 1000}k" if self.n_examples >= 1000 else str(self.n_examples)
        name = f"{self.sampler_type}_N{n_str}_{self.model_type}_b{self.beta}"
        if self.loss_type != "cross_entropy":
            name += f"_{self.loss_type}"
        name += f"_s{self.seed}"
        return name
