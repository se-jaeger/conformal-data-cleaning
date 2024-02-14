import subprocess
from pathlib import Path
from time import sleep

import pandas as pd

from conformal_data_cleaning.config import confidence_levels, error_fractions, method_hyperparameters

BASELINE_EXPERIMENTS = True
CONFORMAL_EXPERIMENTS = True
GARF_EXPERIMENTS = True
EXPERIMENT_BASE_NAME = "final-experiments"
NUM_REPETITIONS = 3
RESULTS_PATH = Path("/conformal-data-cleaning/results") / EXPERIMENT_BASE_NAME
MODELS_PATH = Path("/conformal-data-cleaning/models") / EXPERIMENT_BASE_NAME
HOW_MANY_HPO_TRIALS = 50

###############

helm_install = "helm install --generate-name"
template = str(Path(__file__).parent.parent / "infrastructure/helm/conformal-data-cleaning")
template_garf = str(Path(__file__).parent.parent / "infrastructure/helm/conformal-data-cleaning-garf")
error_fractions_part = f"--set error_fractions='{' '.join([str(x) for x in error_fractions])}'"
num_repetitions_part = f"--set num_repetitions={NUM_REPETITIONS}"
results_path_part = f"--set results_path={RESULTS_PATH}"
models_path_part = f"--set models_path={MODELS_PATH}"
how_many_hpo_trials_part = f"--set how_many_hpo_trials={HOW_MANY_HPO_TRIALS}"

commands: list[str] = []


def baseline_experiments(task_id: int) -> None:
    task_id_part = f"--set task_id={task_id}"

    for method, hyperparameters in method_hyperparameters.items():
        baseline_part = "--set baseline=true"
        method_part = f"--set method={method}"
        for hyperparameter in hyperparameters:
            method_hyperparameters_part = f"--set method_hyperparameter={hyperparameter}"
            commands.append(
                f"{helm_install} {task_id_part} {error_fractions_part} {num_repetitions_part} {models_path_part} {results_path_part} {how_many_hpo_trials_part} {baseline_part} {method_part} {method_hyperparameters_part} {template}",  # noqa: E501
            )


def conformal_experiments(task_id: int) -> None:
    task_id_part = f"--set task_id={task_id}"

    for confidence_level in confidence_levels:
        confidence_level_part = f"--set confidence_level={confidence_level}"

        commands.append(
            f"{helm_install} {task_id_part} {error_fractions_part} {num_repetitions_part} {models_path_part} {results_path_part} {how_many_hpo_trials_part} {confidence_level_part} {template}",  # noqa: E501
        )


# Using 'tabular' datasets following the definition in the paper
dataset_descriptions = pd.read_csv(Path("../data/dataset_descriptions.csv")).convert_dtypes().set_index("task_id")
for task_id in dataset_descriptions[dataset_descriptions["tabular"]].index:
    if BASELINE_EXPERIMENTS:
        baseline_experiments(task_id)

    if CONFORMAL_EXPERIMENTS:
        conformal_experiments(task_id)

    if GARF_EXPERIMENTS:
        task_id_part = f"--set task_id={task_id}"
        commands.append(
            f"{helm_install} {task_id_part} {error_fractions_part} {num_repetitions_part} {models_path_part} {results_path_part} {template_garf}",  # noqa: E501
        )


for command in commands:
    output = subprocess.run(command, shell=True, capture_output=True, check=False)
    print(f"Started: {output.stdout.decode('utf-8').splitlines()[0][6:]}")
    sleep(1)
