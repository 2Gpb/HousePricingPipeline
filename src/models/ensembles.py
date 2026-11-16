import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin

from utils.metrics import rmse


class MeanEnsemble(RegressorMixin, BaseEstimator):
    """Simple average of base models."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanEnsemble":
        """Nothing to learn."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Average of base predictions."""
        return X.mean(axis=1).to_numpy()


class BlendEnsemble(RegressorMixin, BaseEstimator):
    """Weighted average with weights fitted on OOF."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BlendEnsemble":
        """Find non-negative weights that sum to 1 and minimize RMSE."""

        def loss(w: np.ndarray) -> float:
            return rmse(y, X @ w)

        n = X.shape[1]
        x0 = np.full(n, 1 / n)
        bounds = [(0, 1)] * n
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}

        result = minimize(
            loss,
            x0,
            bounds=bounds,
            constraints=constraints,
        )

        self.weights_ = result.x
        print(f"Found weights: {self.weights_}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted sum of base predictions."""
        return (X @ self.weights_).to_numpy()


class StackEnsemble:
    pass
