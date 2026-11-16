from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from configs.config import get_submission_path


def save_test_submission(
    test_pred: np.ndarray, test_ids: pd.Series, config: DictConfig
) -> Path:
    """Save submission with prices in dollars."""
    submission = pd.DataFrame(
        {
            config.data.id_column: test_ids,
            config.data.target_column: test_pred,
        }
    )

    path = get_submission_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return path
