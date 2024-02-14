import json
import sqlite3
import traceback
from argparse import ArgumentParser
from pathlib import Path

import config
import pandas as pd
from att_reverse import att_reverse
from jenga.utils import BINARY_CLASSIFICATION, MULTI_CLASS_CLASSIFICATION, REGRESSION
from rule_sample import rule_sample
from SeqGAN.train import Trainer

from conformal_data_cleaning.config import error_types
from conformal_data_cleaning.jenga_extension import get_OpenMLTask


def run(
    task_id: int,
    error_fractions: list[float],
    num_repetitions: int,
    method: str,
    results_path: Path,
    models_path: Path,
    method_hyperparameter: float = 0.0,  # keep for consistency
) -> None:
    print("Starting baseline with arguments ...")
    print(f"\t--task_id={task_id}")
    print(f"\t--error_fractions={error_fractions}")
    print(f"\t--num_repetitions={num_repetitions}")
    print(f"\t--method={method}")
    print(f"\t--results_path={results_path}")
    print(f"\t--models_path={models_path}")

    for repetition in range(num_repetitions):
        print(f"Start repetition #{repetition + 1} of {num_repetitions}")

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

        ##### GARF #####

        for fraction in error_fractions:
            for corruption in error_types:
                models_base_path = (
                    models_path
                    / str(task_id)
                    / method
                    / str(method_hyperparameter)
                    / corruption
                    / str(fraction)
                    / str(repetition)
                )
                models_base_path.mkdir(parents=True, exist_ok=True)

                g_pre_weights_path = models_base_path / "generator_pre.hdf5"
                d_pre_weights_path = models_base_path / "discriminator_pre.hdf5"
                g_weights_path = models_base_path / "generator.pkl"
                d_weights_path = models_base_path / "discriminator.hdf5"
                path_neg = models_base_path / "generated_sentences.txt"
                path_rules = models_base_path / "rules.txt"

                table_name = f"{task_id}_{corruption}_{fraction}_{repetition}"
                if not Path("database.db").exists():
                    Path("database.db").touch()

                corrupted_task = get_OpenMLTask(task_id=task_id, corruption=corruption, fraction=fraction)

                # prepare and save dirty data into database
                corrupted_task.test_data.astype(str).assign(Label=None).to_sql(
                    table_name,
                    sqlite3.connect("database.db"),
                    if_exists="replace",
                    index=False,
                )

                try:
                    for order in [1, 0]:
                        att_reverse(table_name, order, models_base_path)

                        trainer = Trainer(
                            order=order,
                            B=config.batch_size,
                            T=config.max_length,
                            g_E=config.g_e,
                            g_H=config.g_h,
                            d_E=config.d_e,
                            d_H=config.d_h,
                            d_dropout=config.d_dropout,
                            generate_samples=config.generate_samples,
                            path_pos=table_name,
                            path_neg=path_neg,
                            path_rules=path_rules,
                            g_lr=config.g_lr,
                            d_lr=config.d_lr,
                            n_sample=config.n_sample,
                            models_base_path=models_base_path,
                        )

                        trainer.pre_train(
                            g_epochs=config.g_pre_epochs,
                            d_epochs=config.d_pre_epochs,
                            g_pre_path=g_pre_weights_path,
                            d_pre_path=d_pre_weights_path,
                            g_lr=config.g_pre_lr,
                            d_lr=config.d_pre_lr,
                        )

                        trainer.load_pre_train(g_pre_weights_path, d_pre_weights_path)
                        trainer.reflect_pre_train()  # Mapping layer weights to agent

                        trainer.train(
                            steps=1,
                            g_steps=1,
                            head=10,
                            g_weights_path=g_weights_path,
                            d_weights_path=d_weights_path,
                        )
                        trainer.save(g_weights_path, d_weights_path)

                        rule_len = rule_sample(path_rules, table_name, order)
                        trainer.train_rules(rule_len, path_rules)
                        trainer.filter(table_name)
                        att_reverse(table_name, 1, models_base_path)
                        trainer.repair(table_name)

                except Exception as e:
                    exception_type = str(type(e).__name__)
                    exception_message = str(e)
                    exception_traceback = traceback.format_exc()

                    # Create a dictionary to store the exception information
                    exception_data = {
                        "dataset": table_name,
                        "exception_type": exception_type,
                        "exception_message": exception_message,
                        "exception_traceback": exception_traceback,
                    }

                    # Convert the dictionary to a JSON string
                    json_data = json.dumps(exception_data, indent=4)

                    # Write the JSON string to a text file
                    with open(models_base_path / "ERROR.txt", "w") as file:
                        file.write(json_data)

                    print(exception_traceback)

                ## Evaluation
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

                cleaned_data = (
                    pd.read_sql_query(f"SELECT * FROM '{table_name}'", sqlite3.connect("database.db"))
                    .drop(columns="Label")
                    .convert_dtypes(convert_integer=False, convert_floating=False, convert_boolean=False)
                )

                # more safe to ignore dtypes
                cleaned_mask = corrupted_task.test_data.astype(str) != cleaned_data.astype(str)

                cleaned_perf = original_task.calculate_performance(cleaned_data)
                save_downstream_performance(
                    cleaned_perf,
                    task_type=corrupted_task._task_type,
                    file=directory_for_results / "cleaned_perf.json",
                )

                # save cleaned data
                cleaned_data.to_csv(directory_for_results / "cleaned_data.csv", index=False)
                cleaned_mask.to_csv(directory_for_results / "cleaned_mask.csv", index=False)

        ##### GARF END #####

    (results_path / str(task_id) / method / str(method_hyperparameter) / "finished.txt").touch()


def save_downstream_performance(downstream_metrics: tuple[float, float, float], task_type: int, file: Path) -> None:
    if task_type == BINARY_CLASSIFICATION or task_type == MULTI_CLASS_CLASSIFICATION:
        downstream_metric_names = ("F1_micro", "F1_macro", "F1_weighted")

    elif task_type == REGRESSION:
        downstream_metric_names = ("MAE", "MSE", "RMSE")

    file.parents[0].mkdir(parents=True, exist_ok=True)

    file.write_text(
        json.dumps(
            {
                downstream_metric_names[0]: downstream_metrics[0],  # type: ignore
                downstream_metric_names[1]: downstream_metrics[1],  # type: ignore
                downstream_metric_names[2]: downstream_metrics[2],  # type: ignore
            },
        ),
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="CLI to start a data-cleaning baseline 'garf'.")
    parser.add_argument("--task_id", required=True, type=int)
    parser.add_argument("--error_fractions", required=True, nargs="+", type=float)
    parser.add_argument("--num_repetitions", required=True, type=int)
    parser.add_argument("--results_path", type=Path, required=True)
    parser.add_argument("--models_path", type=Path, required=True)

    parameter_as_dict = {parameter: value for parameter, value in vars(parser.parse_args()).items()}

    run(**parameter_as_dict, method="Garf")
