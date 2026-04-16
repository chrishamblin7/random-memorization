import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_to_gcs(local_dir: str, gcs_path: str):
    """Sync experiment directory to cloud storage.

    Tries the gcsfuse mount at /cloud/misc/ first (fast, no auth needed),
    then falls back to gsutil.
    """
    if not gcs_path:
        return
    cloud_mount = Path("/cloud/misc")
    if cloud_mount.is_dir() and gcs_path.startswith("/cloud/misc/"):
        dst = Path(gcs_path)
        try:
            dst.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["rsync", "-a", "--update", local_dir + "/", str(dst) + "/"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
        return

    try:
        subprocess.run(
            ["gsutil", "-m", "rsync", "-r", local_dir, gcs_path],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: GCS sync failed: {e}")
