"""Deterministic runtime setup and measured environment metadata."""

import importlib.metadata
import os
import platform
import random
import subprocess
import sys

import numpy as np
import psutil
import torch


def deterministic(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def environment():
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_count": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "available_storage_bytes": psutil.disk_usage(".").free,
        "packages": dict(sorted(packages.items())),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "git_commit": git_commit(),
    }


def git_commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None
