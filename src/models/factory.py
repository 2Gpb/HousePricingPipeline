import importlib

import pandas as pd
from catboost import CatBoostRegressor
from omegaconf import DictConfig
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

CONSTRUCTORS = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elastic_net": ElasticNet,
    "knn": KNeighborsRegressor,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    "cat_boost": CatBoostRegressor,
}

# Imported lazily because LightGBM/XGBoost and torch crash together on macOS.
LAZY_CONSTRUCTORS = {
    "lightgbm": ("lightgbm", "LGBMRegressor"),
    "xgboost": ("xgboost", "XGBRegressor"),
}

NETWORKS = {
    "mlp": "MLP",
}

NO_RANDOM_STATE = {"linear_regression", "knn"}


def create_model(config: DictConfig, overrides: dict | None = None) -> BaseEstimator:
    """Regressor from config."""
    name = config.model.name
    known = [*CONSTRUCTORS, *LAZY_CONSTRUCTORS, *NETWORKS]
    if name not in known:
        raise ValueError(f"Unknown model: {name}. Choose from {known}")

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

    if name in NETWORKS:
        from models import architectures
        from models.estimators import NetworkRegressor

        network = getattr(architectures, NETWORKS[name])
        return NetworkRegressor(network, **params)

    if name in LAZY_CONSTRUCTORS:
        module_name, class_name = LAZY_CONSTRUCTORS[name]
        constructor = getattr(importlib.import_module(module_name), class_name)
        return constructor(**params)

    return CONSTRUCTORS[name](**params)


def fit_model(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
    """Fit the model, passing categorical columns to CatBoost."""
    if isinstance(model, CatBoostRegressor):
        cat_features = X.select_dtypes(exclude="number").columns.tolist()
        X[cat_features] = X[cat_features].astype(str).astype(object)
        return model.fit(X, y, cat_features=cat_features)
    return model.fit(X, y)
