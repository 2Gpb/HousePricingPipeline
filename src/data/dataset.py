import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from configs.config import PROJECT_ROOT

ORDINAL_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}


def load_data(config: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train and test."""
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
    """Drop outliers, clean, split features and log target."""
    train = train[
        ~train[config.data.id_column].isin(config.preprocessing.drop_ids)
    ].reset_index(drop=True)

    X = train.drop(
        columns=[
            *config.preprocessing.drop_columns,
            config.data.id_column,
            config.data.target_column,
        ]
    )
    X = _clean(X, config)

    X_test = test.drop(
        columns=[*config.preprocessing.drop_columns, config.data.id_column]
    )
    X_test = _clean(X_test, config)

    y = np.log(train[config.data.target_column])
    test_ids = test[config.data.id_column]

    return X, y, X_test, test_ids


def preprocess_data(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, config: DictConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit on the train fold, apply to all sets."""
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    num_cols = X_train.select_dtypes("number").columns
    cat_cols = X_train.select_dtypes(exclude="number").columns

    cols_fill_mode = list(
        set(
            list(cat_cols)
            + list(config.preprocessing.mode_fill_columns)
            + list(config.preprocessing.ordinal_columns)
        )
    )
    cols_fill_median = num_cols.difference(cols_fill_mode)

    modes = X_train[cols_fill_mode].mode().iloc[0]
    medians = X_train[cols_fill_median].median()

    for X in [X_train, X_val, X_test]:
        X[cols_fill_mode] = X[cols_fill_mode].fillna(modes)
        X[cols_fill_median] = X[cols_fill_median].fillna(medians)

    if config.preprocessing.log_transform:
        for X in [X_train, X_val, X_test]:
            X[list(config.preprocessing.log_columns)] = np.log1p(
                X[list(config.preprocessing.log_columns)]
            )

    if config.preprocessing.encoding == "ohe":
        ohe = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        ).set_output(transform="pandas")
        encoded_train = ohe.fit_transform(X_train[cat_cols])
        encoded_val = ohe.transform(X_val[cat_cols])
        encoded_test = ohe.transform(X_test[cat_cols])

        X_train = pd.concat([X_train.drop(columns=cat_cols), encoded_train], axis=1)
        X_val = pd.concat([X_val.drop(columns=cat_cols), encoded_val], axis=1)
        X_test = pd.concat([X_test.drop(columns=cat_cols), encoded_test], axis=1)
    elif config.preprocessing.encoding == "native":
        for column in cat_cols:
            train_categories = list(X_train[column].unique())
            dtype = pd.CategoricalDtype(categories=[*train_categories, "missing"])
            for frame in (X_train, X_val, X_test):
                known = frame[column].isin(train_categories)
                frame[column] = frame[column].where(known, "missing").astype(dtype)

    if config.preprocessing.scale:
        scaler = StandardScaler().set_output(transform="pandas")
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val[num_cols] = scaler.transform(X_val[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_val, X_test


def _clean(X: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """Rule-based cleaning without statistics."""
    X_to_clean = X.copy()
    cols_as_category = list(config.preprocessing.as_category)
    X_to_clean[cols_as_category] = X_to_clean[cols_as_category].astype(str)

    for part in config.preprocessing.house_parts:
        part_cols = [col for col in X_to_clean.columns if part in col]
        num_cols = X_to_clean[part_cols].select_dtypes("number").columns
        cat_cols = X_to_clean[part_cols].select_dtypes(exclude="number").columns
        no_part = X_to_clean[cat_cols].isna().all(axis=1) & (
            X_to_clean[num_cols].fillna(0) == 0
        ).all(axis=1)

        X_to_clean.loc[no_part, cat_cols] = "None"
        X_to_clean.loc[no_part, num_cols] = 0

    X_to_clean[list(config.preprocessing.unknown_fill_columns)] = X_to_clean[
        list(config.preprocessing.unknown_fill_columns)
    ].fillna("Unknown")

    for col in config.preprocessing.ordinal_columns:
        X_to_clean[col] = X_to_clean[col].map(ORDINAL_MAP)

    return X_to_clean
