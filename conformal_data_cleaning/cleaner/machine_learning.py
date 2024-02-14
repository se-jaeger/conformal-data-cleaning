from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import precision_recall_curve

from ._base import BaseCleaner

logger = getLogger(__name__)


class AutoGluonCleaner(BaseCleaner):
    def __init__(
        self,
        categorical_precision_threshold: float,
        numeric_error_percentile: float,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)

        if categorical_precision_threshold <= 0 or categorical_precision_threshold >= 1:
            raise ValueError(
                "Argument 'categorical_precision_threshold' is not valid! Need to be: 0 <= categorical_precision_threshold <= 1",
            )

        if numeric_error_percentile <= 0 or numeric_error_percentile >= 1:
            raise ValueError(
                "Argument 'numeric_error_percentile' is not valid! Need to be: 0 <= numeric_error_percentile <= 1",
            )

        self._categorical_precision_threshold = categorical_precision_threshold
        self._numeric_error_percentile = numeric_error_percentile

    def __del__(self) -> None:
        for column in self.target_columns_:
            self.predictors_[column].unpersist_models()
            del self.predictors_[column], self.predictors_[column]

        del self.predictors_

    def _fit_method(self, data: pd.DataFrame, **kwargs: dict[str, Any]) -> AutoGluonCleaner:
        self.predictors_: dict[Any, Union[TabularPredictor, TabularPredictor]] = {}

        predictor_params = kwargs.get("predictor_params", {})
        path_prefix = Path(predictor_params.pop("path_prefix", "AutogluonModels"))

        for to_remove in ["label", "quantile_levels"]:
            if predictor_params.pop(to_remove, None) is not None:
                logger.warning(f"Ignoring '{to_remove}' of given 'predictor_params' since it is already defined.")

        for index, column in enumerate(self.target_columns_):
            logger.info(f"Start fitting predictor #{index + 1} of {len(self.target_columns_)}")

            # prepare the path where to store the models
            predictor_params["path"] = path_prefix / str(column)

            # Construction of AutoGluon predictor differs for classification/regression
            # Categorical column => Classification task
            if column in self._categorical_columns:
                is_multi_class = len(data[column].unique()) > 2

                predictor_params["problem_type"] = "multiclass" if is_multi_class else "binary"
                predictor_params["eval_metric"] = "f1_macro" if is_multi_class else "f1"

                self.predictors_[column] = TabularPredictor(label=column, **predictor_params)
                self.predictors_[column].thresholds = {}

            # Numerical column => Regression task
            elif column in self._numerical_columns:
                predictor_params["problem_type"] = "quantile"
                predictor_params["eval_metric"] = "pinball_loss"

                self.predictors_[column] = TabularPredictor(
                    label=column,
                    quantile_levels=[1.0 - self._numeric_error_percentile, 0.5, self._numeric_error_percentile],
                    **predictor_params,
                )

            else:
                raise Exception(f"Column '{column}' is not categorical or numerical ...")

            fit_params = kwargs.get("fit_params", {})

            # Refit/Disk space settings
            fit_params["refit_full"] = True
            fit_params["keep_only_best"] = True
            fit_params["set_best_to_refit_full"] = True
            fit_params["fit_weighted_ensemble"] = False

            # Bagging/Stacking settings
            fit_params["auto_stack"] = False
            fit_params["num_bag_folds"] = 0
            fit_params["num_stack_levels"] = 0

            # HPO settings
            hyperparameter_tune_kwargs = fit_params.pop("hyperparameter_tune_kwargs", {})
            hyperparameter_tune_kwargs["searcher"] = "random"
            hyperparameter_tune_kwargs["scheduler"] = "local"
            hyperparameter_tune_kwargs["num_trials"] = hyperparameter_tune_kwargs.get("num_trials", 10)
            fit_params["hyperparameter_tune_kwargs"] = hyperparameter_tune_kwargs

            self.predictors_[column].fit(data, **fit_params, calibrate=False)

            # After training, we need to optimize precision threshold for categorical columns
            if column in self._categorical_columns:
                probabilities = self.predictors_[column].predict_proba(data)
                for label_idx, label in enumerate(self.predictors_[column].class_labels):
                    precision, _, threshold = precision_recall_curve(
                        (data[column] == label).astype(bool),
                        probabilities.iloc[:, label_idx],
                        pos_label=True,
                    )
                    threshold_for_minimal_precision = threshold[
                        (precision >= self._categorical_precision_threshold).nonzero()[0][0]
                        - 1  # -1 because: len(threshold) == len(rec) -1
                    ]

                    self.predictors_[column].thresholds[label] = threshold_for_minimal_precision

            # save memory during trainng
            self.predictors_[column].unpersist_models()

        return self

    def _remove_outliers_method(self, data: pd.DataFrame, **kwargs: dict[str, Any]) -> pd.DataFrame:
        outliers = {}
        _outlier_predictions = {}

        for column in self.target_columns_:
            predictions = self.predictors_[column].predict(data)

            if column in self._categorical_columns:
                probabilities = self.predictors_[column].predict_proba(data)

                for label_idx, label in enumerate(self.predictors_[column].class_labels):
                    above_precision_predictions = (
                        self.predictors_[column].thresholds[label] <= probabilities.iloc[:, label_idx]
                    )
                    outliers[column] = above_precision_predictions & (data[column] != predictions)

                _outlier_predictions[column] = predictions[outliers[column]]

            elif column in self._numerical_columns:
                outliers[column] = (data[column] < predictions.iloc[:, 0]) | (data[column] > predictions.iloc[:, 2])
                _outlier_predictions[column] = predictions.iloc[outliers[column].values, 1]

            else:
                logger.warning("This should be checked before fit process starts..")

        # calculate all outliers THEN remove them
        # avoid to introduce missing values that need to be handled by the predictors for prediction
        for column in self.target_columns_:
            data.loc[outliers[column], column] = np.nan

        self._outlier_predictions = _outlier_predictions

        return data, None

    def _impute_method(self, data: pd.DataFrame, **kwargs: dict[str, Any]) -> pd.DataFrame:
        for column in self.target_columns_:
            missing_mask = data[column].isna()
            if missing_mask.any():
                predictions = self.predictors_[column].predict(data[missing_mask])

                if column in self._categorical_columns:
                    data.loc[missing_mask, column] = predictions

                elif column in self._numerical_columns:
                    # index 1 is fitted to the 0.5 quantile
                    data.loc[missing_mask, column] = predictions[:, 1]

                else:
                    logger.warning("This should be checked before fit process starts..")

        return data
