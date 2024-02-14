from __future__ import annotations

from logging import getLogger
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from sklearn.preprocessing import OrdinalEncoder

from ._base import BaseCleaner, CleanerError

logger = getLogger(__name__)


class PyodECODCleaner(BaseCleaner):
    def __init__(
        self,
        contamination: float,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)

        if contamination <= 0 or contamination >= 0.5:
            raise ValueError("Argument 'contamination' is not valid! Need to be: 0 <= contamination <= 0.5")

        self.contamination = contamination

    def _fit_method(self, data: pd.DataFrame, **kwargs: dict[str, Any]) -> PyodECODCleaner:
        self.predictors_: dict[Any, ECOD] = {}
        self.encoders_: dict[Any, OrdinalEncoder] = {}
        self._column_means: dict[Any, Union[float, int]] = {}

        for index, column in enumerate(self.target_columns_):
            logger.info(f"Start fitting predictor #{index + 1} of {len(self.target_columns_)}")

            self.predictors_[column] = ECOD(contamination=self.contamination)

            # we need the columns' mean/mode for imputing
            if column in self._categorical_columns:
                self._column_means[column] = data[column].mode()[0]  # multiple modes possible

                # PyOD need preprocessed data, which means we need to encode categorical values
                self.encoders_[column] = OrdinalEncoder(dtype=int, handle_unknown="use_encoded_value", unknown_value=-1)
                column_preprocessed = self.encoders_[column].fit_transform(data[column].to_numpy().reshape(-1, 1))

            elif column in self._numerical_columns:
                self._column_means[column] = data[column].mean()

                column_preprocessed = data[column].to_numpy().reshape(-1, 1)

            else:
                logger.warning("This should be checked before fit process starts..")
                raise CleanerError

            self.predictors_[column].fit(column_preprocessed)  # data need to be two-dimensional

        return self

    def _remove_outliers_method(self, data: pd.DataFrame, **kwargs: dict[str, Any]) -> pd.DataFrame:
        for column in self.target_columns_:
            if column in self._categorical_columns:
                column_preprocessed = self.encoders_[column].transform(data[column].to_numpy().reshape(-1, 1))
            elif column in self._numerical_columns:
                column_preprocessed = data[column].to_numpy().reshape(-1, 1)

            else:
                logger.warning("This should be checked before fit process starts..")
                raise CleanerError

            data.loc[
                self.predictors_[column].predict(column_preprocessed).astype(bool),
                column,
            ] = np.nan

        return data, None

    def _impute_method(self, data: pd.DataFrame, **kwargs: dict[str, Any]) -> pd.DataFrame:
        for column in self.target_columns_:
            missing_mask = data[column].isna()
            if missing_mask.any():
                data.loc[missing_mask, column] = self._column_means[column]

        return data

    # override this to avoid errors.
    def transform(
        self,
        data: pd.DataFrame,
        **kwargs: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        data_without_outliers, outlier_mask, _ = self.remove_outliers(data, **kwargs)

        cleaned_data, imputed_mask = self.impute(data_without_outliers, **kwargs)
        cleaned_mask = imputed_mask | outlier_mask

        return cleaned_data, cleaned_mask, None
