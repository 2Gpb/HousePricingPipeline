import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """RMSE between log prices, as in the Kaggle metric.

    Both y_true and y_pred must already be log(SalePrice).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))
