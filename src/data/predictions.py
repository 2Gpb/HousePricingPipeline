import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from configs.config import PROJECT_ROOT

# Without the same folds and rows the OOF of different experiments can't be combined.
SAME_SETTINGS = [
    "general.seed",
    "validation.n_splits",
    "validation.shuffle",
    "preprocessing.drop_ids",
]


def load_predictions(config: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OOF and test predictions of base experiments as features."""
    experiments_dir = PROJECT_ROOT / config.data.experiments_folder
    experiments = config.model_params[config.model.name].experiments

    for exp in experiments:
        _check_comparable(
            OmegaConf.load(experiments_dir / exp / "config.yml"), exp, config
        )

    X = pd.DataFrame(
        {exp: np.load(experiments_dir / exp / "oof.npy") for exp in experiments}
    )
    X_test = pd.DataFrame(
        {exp: np.load(experiments_dir / exp / "test_pred.npy") for exp in experiments}
    )

    return X, X_test


def _check_comparable(snapshot: DictConfig, name: str, config: DictConfig) -> None:
    """Fail if the experiment used other folds or rows."""
    for key in SAME_SETTINGS:
        if OmegaConf.select(snapshot, key) != OmegaConf.select(config, key):
            raise ValueError(
                f"{name}: {key} = {OmegaConf.select(snapshot, key)}, "
                f"current config has {OmegaConf.select(config, key)}"
            )
