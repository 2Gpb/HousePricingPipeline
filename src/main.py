import numpy as np
from omegaconf import DictConfig

from configs.config import config
from data.dataset import load_data, prepare_data
from models.factory import create_model
from train.validation import run_cv
from utils.log_experiment import check_experiment_is_new, log_experiment, save_artifacts
from utils.submission import save_test_submission


def run(config: DictConfig) -> None:
    """Run CV, save submission, artifacts and log."""
    if not config.general.debug:
        check_experiment_is_new(config)

    train, test = load_data(config)
    X, y, X_test, test_ids = prepare_data(train, test, config)

    model = create_model(config)
    result = run_cv(model, X, y, X_test, config)

    train_mean = np.mean(result["train_scores"])
    val_mean = np.mean(result["val_scores"])
    val_sd = np.std(result["val_scores"])

    metrics = {
        "cv_mean": val_mean,
        "cv_sd": val_sd,
    }

    print(f"Mean train rmse: {train_mean:.5f}")
    print(f"CV rmse: {val_mean:.5f} ± {val_sd:.5f}")

    if not config.general.debug:
        submission_path = save_test_submission(
            np.exp(result["test_pred"]), test_ids, config
        )
        print(f"Submission saved to {submission_path}")
        save_artifacts(result, config)
        log_experiment(metrics, config)


if __name__ == "__main__":
    print(config.general.experiment_name)
    run(config)
