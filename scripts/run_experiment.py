import json
import pickle
from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path
from typing import Union

from jenga.utils import BINARY_CLASSIFICATION, MULTI_CLASS_CLASSIFICATION, REGRESSION

from conformal_data_cleaning import setup_logger
from conformal_data_cleaning.cleaner.conformal import ConformalAutoGluonCleaner
from conformal_data_cleaning.cleaner.machine_learning import AutoGluonCleaner
from conformal_data_cleaning.cleaner.pyod import PyodECODCleaner
from conformal_data_cleaning.config import error_types, method_hyperparameters
from conformal_data_cleaning.jenga_extension import get_OpenMLTask
from conformal_data_cleaning.utils import set_seed

from .hyperparameters import hyperparameters

setup_logger("main")
logger = getLogger("main")
logger.propagate = False


set_seed(42)


def start_baseline(
    task_id: int,
    error_fractions: list[float],
    num_repetitions: int,
    method: str,
    method_hyperparameter: float,
    how_many_hpo_trials: int,
    results_path: Path,
    models_path: Path,
) -> None:
    logger.info("Starting baseline with arguments ...")
    logger.info(f"\t--task_id={task_id}")
    logger.info(f"\t--error_fractions={error_fractions}")
    logger.info(f"\t--num_repetitions={num_repetitions}")
    logger.info(f"\t--method={method}")
    logger.info(f"\t--method_hyperparameter={method_hyperparameter}")
    logger.info(f"\t--how_many_hpo_trials={how_many_hpo_trials}")
    logger.info(f"\t--results_path={results_path}")
    logger.info(f"\t--models_path={models_path}")

    if method not in method_hyperparameters.keys():
        raise ValueError(f"'method' need to be one of: {''.join(method_hyperparameters.keys())}")

    for repetition in range(num_repetitions):
        logger.info(f"Start repetition #{repetition + 1} of {num_repetitions}")

        # get task data and fit baseline model
        original_task = get_OpenMLTask(task_id)
        original_task.fit_baseline_model()
        original_perf = original_task.calculate_performance(original_task.test_data)

        save_downstream_performance(
            original_perf,
            task_type=original_task._task_type,
            file=results_path
            / str(task_id)
            / method
            / str(method_hyperparameter)
            / str(repetition)
            / "original_perf.json",
        )

        # fit cleaning model to clean future incoming potentially dirty data
        if method == "PyodECOD":
            baseline_cleaner: Union[PyodECODCleaner, AutoGluonCleaner] = PyodECODCleaner(
                contamination=method_hyperparameter,
            )
            baseline_cleaner.fit(original_task.train_data)

        elif method == "AutoGluon":
            # prepare path prefix for models
            models_path_prefix = models_path / str(task_id) / method / str(method_hyperparameter) / str(repetition)
            predictor_params = {"path_prefix": models_path_prefix}
            baseline_cleaner = AutoGluonCleaner(
                method_hyperparameter,
                method_hyperparameter,
            )
            baseline_cleaner.fit(
                original_task.train_data,
                fit_params={
                    "verbosity": 1,
                    "hyperparameters": hyperparameters,
                    "hyperparameter_tune_kwargs": {"num_trials": how_many_hpo_trials},
                    "ag_args_fit": {"num_cpus": 24},
                },
                predictor_params=predictor_params,
            )

            # save leaderboards which contains information about training times
            for column, predictor in baseline_cleaner.predictors_.items():
                leaderboard_file_path = f"{models_path_prefix / column}.csv"
                predictor.leaderboard(extra_info=True, silent=True).to_csv(leaderboard_file_path, index=False)

            else:
                logger.warning("This should be checked before..")

        # iterate over corrupted test data
        for fraction in error_fractions:
            for corruption in error_types:
                # downstream performance on corrupted data
                corrupted_task = get_OpenMLTask(task_id=task_id, corruption=corruption, fraction=fraction)
                corrupted_perf = original_task.calculate_performance(corrupted_task.test_data)

                directory_for_results = (
                    results_path
                    / str(task_id)
                    / method
                    / str(method_hyperparameter)
                    / str(repetition)
                    / corruption
                    / str(fraction)
                )
                directory_for_results.mkdir(parents=True, exist_ok=True)

                save_downstream_performance(
                    corrupted_perf,
                    task_type=corrupted_task._task_type,
                    file=directory_for_results / "corrupted_perf.json",
                )

                cleaned_data, cleaned_mask, _ = baseline_cleaner.transform(
                    corrupted_task.test_data,
                )

                cleaned_perf = original_task.calculate_performance(cleaned_data)
                save_downstream_performance(
                    cleaned_perf,
                    task_type=corrupted_task._task_type,
                    file=directory_for_results / "cleaned_perf.json",
                )

                # save cleaned data
                cleaned_data.to_csv(directory_for_results / "cleaned_data.csv", index=False)
                cleaned_mask.to_csv(directory_for_results / "cleaned_mask.csv", index=False)

        # cleanup before next iteration
        del baseline_cleaner, original_task, corrupted_task

    # indicate that this experiment is already finished
    (results_path / str(task_id) / method / str(method_hyperparameter) / "finished.txt").touch()


def start_experiment(
    task_id: int,
    confidence_level: float,
    error_fractions: list[float],
    num_repetitions: int,
    how_many_hpo_trials: int,
    results_path: Path,
    models_path: Path,
) -> None:
    logger.info("Starting experiment with arguments ...")
    logger.info(f"\t--task_id={task_id}")
    logger.info(f"\t--confidence_level={confidence_level}")
    logger.info(f"\t--error_fractions={error_fractions}")
    logger.info(f"\t--num_repetitions={num_repetitions}")
    logger.info(f"\t--how_many_hpo_trials={how_many_hpo_trials}")
    logger.info(f"\t--results_path={results_path}")
    logger.info(f"\t--models_path={models_path}")

    for repetition in range(num_repetitions):
        logger.info(f"Start repetition #{repetition + 1} of {num_repetitions}")

        # get task data and fit baseline model
        original_task = get_OpenMLTask(task_id)
        original_task.fit_baseline_model()
        original_perf = original_task.calculate_performance(original_task.test_data)

        save_downstream_performance(
            original_perf,
            task_type=original_task._task_type,
            file=results_path
            / str(task_id)
            / "ConformalAutoGluon"
            / str(confidence_level)
            / str(repetition)
            / "original_perf.json",
        )

        # prepare path prefix for models
        models_path_prefix = models_path / str(task_id) / "ConformalAutoGluon" / str(confidence_level) / str(repetition)
        ci_ag_predictor_params = {"path_prefix": models_path_prefix}

        # fit cleaning model to clean future incoming potentially dirty data
        cleaner = ConformalAutoGluonCleaner(
            confidence_level=confidence_level,
        )
        cleaner.fit(
            original_task.train_data,
            ci_ag_fit_params={
                "verbosity": 1,
                "hyperparameters": hyperparameters,
                "hyperparameter_tune_kwargs": {"num_trials": how_many_hpo_trials},
                "ag_args_fit": {"num_cpus": 24},
            },
            ci_ag_predictor_params=ci_ag_predictor_params,
        )

        # save leaderboards which contains information about training times
        for column, predictor in cleaner.predictors_.items():
            leaderboard_file_path = f"{models_path_prefix / column}.csv"
            predictor._predictor.leaderboard(extra_info=True, silent=True).to_csv(leaderboard_file_path, index=False)

        # iterate over corrupted test data
        for fraction in error_fractions:
            for corruption in error_types:
                # downstream performance on corrupted data
                corrupted_task = get_OpenMLTask(task_id=task_id, corruption=corruption, fraction=fraction)
                corrupted_perf = original_task.calculate_performance(corrupted_task.test_data)

                directory_for_results = (
                    results_path
                    / str(task_id)
                    / "ConformalAutoGluon"
                    / str(confidence_level)
                    / str(repetition)
                    / corruption
                    / str(fraction)
                )
                directory_for_results.mkdir(parents=True, exist_ok=True)

                save_downstream_performance(
                    corrupted_perf,
                    task_type=corrupted_task._task_type,
                    file=directory_for_results / "corrupted_perf.json",
                )

                # downstream performance on cleaned data
                cleaned_data, cleaned_mask, prediction_sets = cleaner.transform(
                    corrupted_task.test_data,
                    empty_pred_sets_are_inliers=True,  # type: ignore
                )

                with open(directory_for_results / "prediction_sets.pckl", "wb") as file:
                    pickle.dump(prediction_sets, file)

                cleaned_perf = original_task.calculate_performance(cleaned_data)
                save_downstream_performance(
                    cleaned_perf,
                    task_type=corrupted_task._task_type,
                    file=directory_for_results / "cleaned_perf.json",
                )

                # save cleaned data
                cleaned_data.to_csv(directory_for_results / "cleaned_data.csv", index=False)
                cleaned_mask.to_csv(directory_for_results / "cleaned_mask.csv", index=False)

        # cleanup before next iteration
        del cleaner, original_task, corrupted_task

    # indicate that this experiment is already finished
    (results_path / str(task_id) / "ConformalAutoGluon" / str(confidence_level) / "finished.txt").touch()


def save_downstream_performance(downstream_metrics: tuple[float, float, float], task_type: int, file: Path) -> None:
    if task_type == BINARY_CLASSIFICATION or task_type == MULTI_CLASS_CLASSIFICATION:
        downstream_metric_names = ("F1_micro", "F1_macro", "F1_weighted")

    elif task_type == REGRESSION:
        downstream_metric_names = ("MAE", "MSE", "RMSE")

    file.parents[0].mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results at: {file}")
    file.write_text(
        json.dumps(
            {
                downstream_metric_names[0]: downstream_metrics[0],  # type: ignore
                downstream_metric_names[1]: downstream_metrics[1],  # type: ignore
                downstream_metric_names[2]: downstream_metrics[2],  # type: ignore
            },
        ),
    )


def start() -> None:
    """CLI implementation of the `run-experiment` command."""
    parser = ArgumentParser(description="CLI to start a data-cleaning experiment.")
    parser.add_argument("--task_id", required=True, type=int)
    parser.add_argument("--error_fractions", required=True, nargs="+", type=float)
    parser.add_argument("--num_repetitions", required=True, type=int)
    parser.add_argument("--results_path", type=Path, required=True)
    parser.add_argument("--models_path", type=Path, required=True)
    parser.add_argument("--how_many_hpo_trials", required=True, type=int)

    subparsers = parser.add_subparsers()

    # baseline
    baseline_parser = subparsers.add_parser("baseline")
    baseline_parser.set_defaults(command_function=start_baseline)

    baseline_parser.add_argument("--method", required=True, type=str)
    baseline_parser.add_argument("--method_hyperparameter", required=True, type=float)

    # experiment
    experiment_parser = subparsers.add_parser("experiment")
    experiment_parser.set_defaults(command_function=start_experiment)

    experiment_parser.add_argument("--confidence_level", required=True, type=float)

    # RUN!
    args = parser.parse_args()
    args.command_function(
        # call given command function with parameters
        **{parameter: value for parameter, value in vars(args).items() if parameter != "command_function"},
    )
