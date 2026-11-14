import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from omegaconf import DictConfig
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

CONSTRUCTORS = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elastic_net": ElasticNet,
    "knn": KNeighborsRegressor,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    "cat_boost": CatBoostRegressor,
    "lightgbm": LGBMRegressor,
    "xgboost": XGBRegressor,
}

NO_RANDOM_STATE = {"linear_regression", "knn"}


def create_model(config: DictConfig, overrides: dict | None = None) -> BaseEstimator:
    """Create the configured regressor."""
    name = config.model.name
    if name not in CONSTRUCTORS:
        raise ValueError(f"Unknown model: {name}. Choose from {list(CONSTRUCTORS)}")

    params = dict(config.model_params.get(name) or {})
    if overrides:
        params.update(overrides)

    if name not in NO_RANDOM_STATE:
        params.setdefault("random_state", config.general.seed)
    if name == "cat_boost":
        params.setdefault("verbose", False)
        params.setdefault("allow_writing_files", False)
    if name == "lightgbm":
        params.setdefault("verbose", -1)
    if name == "xgboost":
        params.setdefault("enable_categorical", True)
        params.setdefault("tree_method", "hist")

    return CONSTRUCTORS[name](**params)


def fit_model(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
    """Fit the model; CatBoost also needs the list of categorical columns."""
    if isinstance(model, CatBoostRegressor):
        cat_features = X.select_dtypes(exclude="number").columns.tolist()
        X[cat_features] = X[cat_features].astype(str).astype(object)
        return model.fit(X, y, cat_features=cat_features)
    return model.fit(X, y)
