from pathlib import Path

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]

config_path = Path(__file__).parent / "config.yml"
config = OmegaConf.load(config_path)


def get_submission_path(config: DictConfig) -> Path:
    """Submission file path."""
    return (
        PROJECT_ROOT
        / config.data.submission_folder
        / f"{config.general.experiment_name}.csv"
    )


def get_experiment_dir(config: DictConfig) -> Path:
    """Experiment artifacts folder."""
    return (
        PROJECT_ROOT / config.data.experiments_folder / config.general.experiment_name
    )


def get_log_path(config: DictConfig) -> Path:
    """Experiment log path."""
    return PROJECT_ROOT / config.data.experiments_folder / config.data.log_file
