import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class MeanEnsemble(RegressorMixin, BaseEstimator):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanEnsemble":
        """Nothing to learn."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Average of base predictions."""
        return X.mean(axis=1).to_numpy()


class BlendEnsemble:
    pass


class StackEnsemble:
    pass
