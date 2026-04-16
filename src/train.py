import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from .config import ExperimentConfig
from .data import RandomMemorizationData, build_sampler
from .losses import stablemax_cross_entropy
from .model import build_model
from .utils import set_seed, sync_to_gcs


EVAL_BATCH_SIZE = 4096


def _compute_loss(logits: torch.Tensor, targets: torch.Tensor, loss_type: str,
                  reduction: str = "mean") -> torch.Tensor:
    """Compute loss over output positions.

    logits:  (B, output_len, output_vocab_size)
    targets: (B, output_len)
    """
    B, L, V = logits.shape
    logits_flat = logits.reshape(B * L, V)
    targets_flat = targets.reshape(B * L)
    if loss_type == "stablemax":
        return stablemax_cross_entropy(logits_flat, targets_flat, reduction=reduction)
    return F.cross_entropy(logits_flat, targets_flat, reduction=reduction)


def evaluate(model, data: RandomMemorizationData, device: str,
             loss_type: str = "cross_entropy"):
    """Evaluate on ALL examples. Returns (per_example, agg_acc, agg_loss)."""
    n = data.n_examples
    output_len = data.output_len

    per_loss = torch.zeros(n)
    per_correct = torch.zeros(n, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        for start in range(0, n, EVAL_BATCH_SIZE):
            end = min(start + EVAL_BATCH_SIZE, n)
            xb = data.inputs[start:end].to(device)
            tb = data.targets[start:end].to(device)

            logits = model(xb)
            B, L, V = logits.shape
            loss_per = _compute_loss(logits, tb, loss_type, reduction="none")
            loss_per = loss_per.view(B, L).mean(dim=1)
            preds = logits.argmax(dim=-1)
            correct = (preds == tb).all(dim=-1).long()

            per_loss[start:end] = loss_per.cpu()
            per_correct[start:end] = correct.cpu()

    model.train()

    per_example = {}
    for i in range(n):
        per_example[i] = {
            "loss": per_loss[i].item(),
            "acc": per_correct[i].item(),
        }

    agg_acc = per_correct.float().mean().item()
    agg_loss = per_loss.mean().item()
    return per_example, agg_acc, agg_loss


def train(cfg: ExperimentConfig):
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    run_name = cfg.auto_run_name()
    run_dir = Path("results") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(str(run_dir / "config.yaml"))

    wandb_kwargs = dict(project=cfg.wandb_project, name=run_name, config={
        k: v for k, v in vars(cfg).items() if not k.startswith("_")
    })
    if cfg.resume:
        wandb_kwargs["resume"] = "allow"
    wandb.init(**wandb_kwargs)

    data = RandomMemorizationData(cfg)
    sampler = build_sampler(cfg)

    with open(run_dir / "sampler_probs.json", "w") as f:
        json.dump({str(i): p for i, p in sampler.prob_of.items()}, f)

    print(f"=== {run_name} ===")
    print(f"Examples: {cfg.n_examples}")
    print(f"Input   : len={cfg.input_len}, vocab={cfg.input_vocab_size}")
    print(f"Output  : len={cfg.output_len}, vocab={cfg.output_vocab_size}")
    print(f"Model   : {cfg.model_type}")
    print(f"Sampler : {sampler.name()}")
    print(f"Loss    : {cfg.loss_type}")
    print(f"Device  : {device}")

    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params  : {n_params:,}")
    wandb.log({"n_params": n_params}, step=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    rng = np.random.default_rng(cfg.seed + 1)
    metrics_log: list[dict] = []
    start_step = 1

    if cfg.resume:
        start_step, metrics_log = _try_resume(model, optimizer, run_dir)

    t0 = time.time()
    try:
        for step in range(start_step, cfg.num_steps + 1):
            x, targets = data.sample_batch(sampler, rng, cfg.batch_size, device)
            logits = model(x)
            loss = _compute_loss(logits, targets, cfg.loss_type)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                wandb.log({"train/loss": loss.item()}, step=step)

            if step % cfg.eval_every == 0:
                per_example, agg_acc, agg_loss = evaluate(model, data, device, cfg.loss_type)

                log_dict = {"eval/loss": agg_loss, "eval/acc": agg_acc}
                wandb.log(log_dict, step=step)

                entry = {
                    "step": step,
                    "train_loss": loss.item(),
                    "agg_acc": agg_acc,
                    "agg_loss": agg_loss,
                    "per_example": {str(i): v for i, v in per_example.items()},
                }
                metrics_log.append(entry)

                elapsed = time.time() - t0
                print(
                    f"[{elapsed:7.1f}s] step {step:>6d}  "
                    f"train_loss={loss.item():.4f}  "
                    f"acc={agg_acc:.4f}  loss={agg_loss:.4f}"
                )

                _save_metrics(metrics_log, run_dir)
                _regenerate_plots(run_dir)

            if step % cfg.checkpoint_every == 0:
                ckpt_dir = run_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                torch.save(
                    {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                    ckpt_dir / f"step_{step}.pt",
                )
                _save_metrics(metrics_log, run_dir)
                sync_to_gcs(str(run_dir), cfg.gcs_bucket + run_name)

    except KeyboardInterrupt:
        print("\nInterrupted -- saving checkpoint...")

    _save_metrics(metrics_log, run_dir)
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    sync_to_gcs(str(run_dir), cfg.gcs_bucket + run_name)
    wandb.finish()
    print(f"Done. Results in {run_dir}")


def _try_resume(model, optimizer, run_dir):
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        print("[resume] No checkpoints directory found, starting from scratch")
        return 1, []

    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        print("[resume] No checkpoint files found, starting from scratch")
        return 1, []

    latest = ckpts[-1]
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt["step"]
    print(f"[resume] Restored from {latest.name} (step {step})")

    metrics_log = []
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics_log = json.load(f)
        metrics_log = [e for e in metrics_log if e["step"] <= step]
        print(f"[resume] Loaded {len(metrics_log)} metric entries up to step {step}")

    return step + 1, metrics_log


def _save_metrics(metrics_log, run_dir):
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f)


def _regenerate_plots(run_dir: Path):
    script = Path(__file__).resolve().parent.parent / "scripts" / "plot_loss_curves.py"
    if not script.exists():
        return
    try:
        subprocess.Popen(
            [sys.executable, str(script), "--results-dir", str(run_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
