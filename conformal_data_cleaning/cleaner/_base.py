from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.utils.validation import check_is_fitted

from ..utils import is_categorical, set_seed


class CleanerError(Exception):
    """Exception raised for errors in Imputers."""


class BaseCleaner(ABC):
    _outlier_predictions: dict

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        set_seed(self._seed)

    def _guess_dtypes(self, data: pd.DataFrame) -> None:
        self._categorical_columns = [c for c in data.columns if is_categorical(data[c])]
        self._numerical_columns = [
            c for c in data.columns if is_numeric_dtype(data[c]) and c not in self._categorical_columns
        ]

        if len(data.columns) != (len(self._categorical_columns) + len(self._numerical_columns)):
            raise Exception(
                f"There are {len(data.columns)} columns but found "
                f"{len(self._categorical_columns)} categorical and "
                f"{len(self._numerical_columns)} numerical columns.",
            )

    def fit(self, data: pd.DataFrame, target_columns: Optional[list] = None, **kwargs: dict[str, Any]) -> BaseCleaner:
        if target_columns is None:
            target_columns = data.columns.to_list()

        if not type(target_columns) == list:
            raise CleanerError(
                f"Parameter 'target_column' need to be of type list\
                    but is '{type(target_columns)}'",
            )

        if any([column not in data.columns for column in target_columns]):
            raise CleanerError(f"All target columns ('{target_columns}') must be in: {', '.join(data.columns)}")

        self.target_columns_ = target_columns

        self._guess_dtypes(data)
        return self._fit_method(data=data.copy(), **kwargs)

    def remove_outliers(
        self,
        data: pd.DataFrame,
        **kwargs: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        check_is_fitted(self, ["predictors_", "target_columns_"])

        missing_mask = data[self.target_columns_].isna()
        data_without_outliers, prediction_sets = self._remove_outliers_method(data=data.copy(), **kwargs)

        missing_mask_outliers_removed = data_without_outliers[self.target_columns_].isna()
        outlier_mask = missing_mask_outliers_removed & ~missing_mask

        return data_without_outliers, outlier_mask, prediction_sets

    def impute(self, data: pd.DataFrame, **kwargs: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
        check_is_fitted(self, ["predictors_", "target_columns_"])

        missing_mask = data[self.target_columns_].isna()
        imputed_data = self._impute_method(data=data.copy(), **kwargs)

        return imputed_data, missing_mask

    def transform(
        self,
        data: pd.DataFrame,
        **kwargs: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        data_without_outliers, outlier_mask, prediction_sets = self.remove_outliers(data, **kwargs)

        if kwargs.get("reuse_intermediate", True):
            for column in self.target_columns_:
                data_without_outliers.loc[outlier_mask.loc[:, column], column] = self._outlier_predictions[column]

        cleaned_data, imputed_mask = self.impute(data_without_outliers, **kwargs)
        cleaned_mask = imputed_mask | outlier_mask

        delattr(self, "_outlier_predictions")

        return cleaned_data, cleaned_mask, prediction_sets

    @abstractmethod
    def _fit_method(self, data: pd.DataFrame, **kwargs: dict[str, Any]) -> BaseCleaner:
        pass

    @abstractmethod
    def _remove_outliers_method(
        self,
        data: pd.DataFrame,
        **kwargs: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass

    @abstractmethod
    def _impute_method(self, data: pd.DataFrame, **kwargs: dict[str, Any]) -> pd.DataFrame:
        pass
