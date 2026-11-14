import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from configs.config import (
    PROJECT_ROOT,
    get_experiment_dir,
    get_log_path,
    get_submission_path,
)


def check_experiment_is_new(config: DictConfig) -> None:
    """Fail before training if this experiment name was already used."""
    existing = [
        path
        for path in (get_experiment_dir(config), get_submission_path(config))
        if path.exists()
    ]
    if existing:
        paths = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            f"Experiment '{config.general.experiment_name}' already exists:\n{paths}\n"
            "Change general.experiment_name in config.yml."
        )


def save_artifacts(result: dict, config: DictConfig) -> None:
    """Save the config snapshot, OOF and test predictions (log space)."""
    directory = get_experiment_dir(config)
    directory.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config, directory / "config.yml")
    np.save(directory / "oof.npy", result["oof"])
    np.save(directory / "test_pred.npy", result["test_pred"])


def log_experiment(metrics: dict, config: DictConfig) -> None:
    """Append one row with metrics and git state to the experiment log."""
    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "experiment_name": config.general.experiment_name,
                **_get_git_info(),
                "model": config.model.name,
                "cv_mean": metrics["cv_mean"],
                "cv_sd": metrics["cv_sd"],
                "public": "",
            }
        ]
    )

    path = get_log_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def _get_git_info() -> dict:
    """Return the current commit hash and whether tracked files have uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        ).stdout.strip()

    return {
        "git_hash": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }
