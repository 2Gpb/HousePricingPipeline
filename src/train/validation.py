import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold

from data.dataset import preprocess_data
from models.factory import fit_model
from utils.metrics import rmse


def run_cv(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    config: DictConfig,
) -> dict:
    """K-fold CV: fold scores, OOF and averaged test predictions."""
    cv = KFold(
        n_splits=config.validation.n_splits,
        shuffle=config.validation.shuffle,
        random_state=config.general.seed,
    )

    train_scores, val_scores, test_preds = [], [], []
    oof = np.full(len(X), np.nan)
    for fold, (train_idx, val_idx) in enumerate(cv.split(X), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        X_train, X_val, fold_test = preprocess_data(X_train, X_val, X_test, config)

        fold_model = clone(model)
        fit_model(fold_model, X_train, y_train)

        train_pred = fold_model.predict(X_train)
        val_pred = fold_model.predict(X_val)
        test_pred = fold_model.predict(fold_test)

        train_scores.append(rmse(y_train, train_pred))
        val_scores.append(rmse(y_val, val_pred))
        oof[val_idx] = val_pred
        test_preds.append(test_pred)

        print(f"Fold {fold}: train={train_scores[-1]:.5f}, val={val_scores[-1]:.5f}")

    return {
        "train_scores": train_scores,
        "val_scores": val_scores,
        "test_pred": np.mean(test_preds, axis=0),
        "oof": oof,
    }
