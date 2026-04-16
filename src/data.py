from abc import ABC, abstractmethod

import numpy as np
import torch


class ExampleSampler(ABC):
    prob_of: dict[int, float]

    @abstractmethod
    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class PowerLawExampleSampler(ExampleSampler):
    """P(example i) proportional to (i+1)^{-beta}."""

    def __init__(self, n_examples: int, beta: float = 1.5):
        self.n_examples = n_examples
        self.beta = beta
        indices = np.arange(n_examples)
        w = (indices + 1.0) ** (-beta)
        self.probs = w / w.sum()
        self.prob_of = {i: float(p) for i, p in enumerate(self.probs)}

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.choice(self.n_examples, size=n, p=self.probs)

    def name(self) -> str:
        return f"power_law_beta{self.beta}"


class UniformExampleSampler(ExampleSampler):
    def __init__(self, n_examples: int):
        self.n_examples = n_examples
        p = 1.0 / n_examples
        self.prob_of = {i: p for i in range(n_examples)}

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.integers(0, self.n_examples, size=n)

    def name(self) -> str:
        return "uniform"


class RandomMemorizationData:
    """Static dataset of random input-output pairs.

    Each example i has:
      - inputs[i]: shape (input_len,) of random tokens from {0, ..., input_vocab_size-1}
      - targets[i]: shape (output_len,) of random tokens from {0, ..., output_vocab_size-1}

    The model sees the full input and must predict targets for the last output_len positions.
    """

    def __init__(self, cfg):
        self.n_examples = cfg.n_examples
        self.input_len = cfg.input_len
        self.output_len = cfg.output_len
        self.input_vocab_size = cfg.input_vocab_size
        self.output_vocab_size = cfg.output_vocab_size

        rng = np.random.default_rng(cfg.data_seed)

        inputs = rng.integers(0, cfg.input_vocab_size, size=(cfg.n_examples, cfg.input_len))

        seen = set()
        for i in range(cfg.n_examples):
            key = inputs[i].tobytes()
            while key in seen:
                inputs[i] = rng.integers(0, cfg.input_vocab_size, size=cfg.input_len)
                key = inputs[i].tobytes()
            seen.add(key)

        targets = rng.integers(0, cfg.output_vocab_size, size=(cfg.n_examples, cfg.output_len))

        self.inputs = torch.from_numpy(inputs.astype(np.int64))
        self.targets = torch.from_numpy(targets.astype(np.int64))

    def sample_batch(
        self,
        sampler: ExampleSampler,
        rng: np.random.Generator,
        batch_size: int,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = sampler.sample(rng, batch_size)
        return (
            self.inputs[indices].to(device),
            self.targets[indices].to(device),
        )


def build_sampler(cfg) -> ExampleSampler:
    if cfg.sampler_type == "power_law":
        return PowerLawExampleSampler(cfg.n_examples, cfg.beta)
    if cfg.sampler_type == "uniform":
        return UniformExampleSampler(cfg.n_examples)
    raise ValueError(f"Unknown sampler_type: {cfg.sampler_type}")
