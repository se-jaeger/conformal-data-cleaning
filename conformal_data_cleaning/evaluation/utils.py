import warnings
from typing import Any

import pandas as pd
from sklearn.preprocessing import minmax_scale


class AutoInitializedDict(dict):
    def __getitem__(self, item: Any) -> Any:
        try:
            return dict.__getitem__(self, item)

        except KeyError:
            value = self[item] = type(self)()
            return value


def calculate_percent_improvements(df: pd.DataFrame) -> float:
    return round(df["improvement"].mean() * 100, 2)


def calculate_median_percent_error_detection(df: pd.DataFrame) -> float:
    return round(df["error_detection_fraction__mean"].median() * 100, 2)


def calculate_median_percent_error_wrong_detection(df: pd.DataFrame) -> float:
    return round(df["error_wrong_detection_fraction__mean"].median() * 100, 2)


def calculate_median_percent_numerical_error_reduction(df: pd.DataFrame) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if df["num_numerical"].unique()[0] > 0:
            error_reduction_relative = df["numerical_error_reduction__mean"] / df["numerical_error"]
            mean_error_reduction_relative = error_reduction_relative.median()

            return round(mean_error_reduction_relative * 100, 2)
        else:
            return pd.NA


def calculate_median_percent_categorical_error_reduction(df: pd.DataFrame) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if df["num_categorical"].unique()[0] > 0:
            error_reduction_relative = df["categorical_error_reduction__mean"] / df["categorical_error"]
            mean_error_reduction_relative = error_reduction_relative.median()

            return round(mean_error_reduction_relative * 100, 2)
        else:
            return pd.NA


def calculate_median_percent_categorical_error_reduction_weighted(df: pd.DataFrame) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if df["num_categorical"].unique()[0] > 0:
            error_reduction_relative = (
                df["categorical_error_weighted_reduction__mean"] / df["categorical_error_weighted"]
            )
            error_reduction_relative.loc[(df["categorical_error"] == 1)] = pd.NA
            mean_error_reduction_relative = error_reduction_relative.median()

            return round(mean_error_reduction_relative * 100, 2)
        else:
            return pd.NA


def normalize_performance(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["corrupted_performance__mean", "cleaned_performance__mean"]

    # depending on the downstream task and, therefore, the metric RMSE vs. F1
    # we use different 'feature_range's. Later on, we use the absolute values
    # to get values between 0 (worst) and 1 (best).
    if df["task_type"].unique()[0] == "regression":
        feature_range = (-1, 0)
    else:
        feature_range = (0, 1)

    result_as_dict = {column: minmax_scale(df[column], feature_range=feature_range) for column in columns}
    return pd.DataFrame(result_as_dict, index=df.index).rename(columns=lambda x: f"{x}_normalized").abs()


def normalize_improvement(df: pd.DataFrame) -> pd.DataFrame:
    column = "improvement_in_percent__mean"
    improvement_mask = df[column] >= 0
    result = pd.DataFrame(index=df.index)

    # preserve whether there is an improvement or decrease in performance.
    if len(df.loc[improvement_mask, column]) != 0:
        result.loc[improvement_mask, column] = minmax_scale(df.loc[improvement_mask, column], feature_range=(0, 1))

    if len(df.loc[~improvement_mask, column]) != 0:
        result.loc[~improvement_mask, column] = minmax_scale(df.loc[~improvement_mask, column], feature_range=(-1, 0))

    return result.rename(columns=lambda x: f"{x}_normalized")
