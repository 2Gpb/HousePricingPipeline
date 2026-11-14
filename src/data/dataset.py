import numpy as np
import pandas as pd
from omegaconf import DictConfig

from configs.config import PROJECT_ROOT


def load_data(config: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train and test CSVs, keeping "None" as a real category."""
    train = pd.read_csv(
        PROJECT_ROOT / config.data.train_path,
        keep_default_na=False,
        na_values=["NA"],
    )
    test = pd.read_csv(
        PROJECT_ROOT / config.data.test_path,
        keep_default_na=False,
        na_values=["NA"],
    )
    return train, test


def prepare_data(
    train: pd.DataFrame, test: pd.DataFrame, config: DictConfig
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Drop outliers and split into features, log target, test features and test ids."""
    train = train[~train[config.data.id_column].isin(config.data.drop_ids)].reset_index(
        drop=True
    )
    X = train.drop(columns=[config.data.id_column, config.data.target_column])
    y = np.log(train[config.data.target_column])
    X_test = test.drop(columns=[config.data.id_column])
    test_ids = test[config.data.id_column]
    return X, y, X_test, test_ids


def preprocess_data(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, config: DictConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit preprocessing on the train fold and apply it to all three sets."""
    cols = X_train.select_dtypes("number").columns
    medians = X_train[cols].median()

    X_train = X_train[cols].fillna(medians)
    X_val = X_val[cols].fillna(medians)
    X_test = X_test[cols].fillna(medians)

    return X_train, X_val, X_test
