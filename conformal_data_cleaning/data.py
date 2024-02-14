from enum import Enum
from json import loads
from logging import getLogger
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

logger = getLogger()

_DATA_BASE_PATH = Path(__file__).parent.parent / "data"  # Local path
_TRAINING_PATH = _DATA_BASE_PATH / "training"
_TEST_PATH = _DATA_BASE_PATH / "test"
_CORRUPTED_PATH = _DATA_BASE_PATH / "corrupted"
_IDS_FILE_POST_FIX = "openml_ids.json"
_ID_TO_TASK_TYPE = {}


class TaskType(str, Enum):
    BINARY = "binary"
    MULTI_CLASS = "multi_class"
    REGRESSION = "regression"


class CorruptionType(str, Enum):
    SCALING = "Scaling"
    GAUSSIAN_NOISE = "GaussianNoise"
    SWAPPED_VALUES = "SwappedValues"
    CATEGORICAL_SHIFT = "CategoricalShift"


for task_type in TaskType:
    with open(_DATA_BASE_PATH / f"{task_type.value}_{_IDS_FILE_POST_FIX}") as file:
        for task_id in loads(file.read()):
            _ID_TO_TASK_TYPE[int(task_id)] = task_type


AVAILABLE_DATASETS = list(_ID_TO_TASK_TYPE.keys())


def get_X_y_paths(
    task_id: int,
    training: bool = True,
    corruption: Optional[str] = None,
    fraction: Optional[float] = None,
) -> tuple[Path, Path]:
    if (corruption is not None and fraction is None) or (corruption is None and fraction is not None):
        raise Exception("Either set 'corruption' and 'fraction' or none of them!")

    if corruption is not None and fraction is not None:
        if corruption not in CorruptionType.__members__.values():
            raise Exception(f"Given value for 'corruption' ({corruption}) is not valid!")
        path_prefix = _CORRUPTED_PATH / corruption / str(fraction)

    else:
        path_prefix = _TRAINING_PATH if training else _TEST_PATH

    X_path = path_prefix / f"{task_id}_X.csv"
    y_path = path_prefix / f"{task_id}_y.csv"

    return X_path, y_path


def fetch_and_save_dataset(task_id: int, training_size: float = 0.8) -> bool:
    try:
        if task_id not in AVAILABLE_DATASETS:
            raise ValueError(f"Dataset with ID {task_id} not available.")

        X, y = fetch_openml(data_id=task_id, as_frame=True, return_X_y=True)

        # stratify split for classification tasks
        if _ID_TO_TASK_TYPE[task_id] == TaskType.BINARY or _ID_TO_TASK_TYPE[task_id] == TaskType.MULTI_CLASS:
            args = {"stratify": y}

        else:
            args = {}

        training_X, test_X, training_y, test_y = train_test_split(
            X,
            y,
            train_size=training_size,
            **args,
            random_state=42,  # make reproducible
        )

        # save both (training and test) sets
        for is_training, X_temp, y_temp in zip([True, False], [training_X, test_X], [training_y, test_y]):
            X_path, y_path = get_X_y_paths(task_id=task_id, training=is_training)

            X_temp.to_csv(X_path, index=False)
            y_temp.to_csv(y_path, index=False)

            assert X_path.exists() and X_path.is_file()
            assert y_path.exists() and y_path.is_file()

    except Exception as error:  # noqa
        return False

    return True


def read_dataset(
    task_id: int,
    training: bool = True,
    corruption: Optional[str] = None,
    fraction: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if (corruption is not None and fraction is None) or (corruption is None and fraction is not None):
        raise Exception("Either set 'corruption' and 'fraction' or none of them!")

    if task_id not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset with ID {task_id} not available.")

    X_path, y_path = get_X_y_paths(task_id=task_id, training=training, corruption=corruption, fraction=fraction)

    # check if dataset is available
    if not (X_path.exists() and X_path.is_file()) or not (y_path.exists() and y_path.is_file()):
        raise Exception(f"Dataset with ID {task_id} is not downloaded yet. Use 'fetch_and_save_dataset' first.")

    # AutoGluon can't handle the "new" data types.
    # For this reason, we infer objects but do not cast further
    X: pd.DataFrame = (
        pd.read_csv(X_path)
        .squeeze("columns")
        .convert_dtypes(convert_integer=False, convert_floating=False, convert_boolean=False)
    )
    y: pd.Series = (
        pd.read_csv(y_path)
        .squeeze("columns")
        .convert_dtypes(convert_integer=False, convert_floating=False, convert_boolean=False)
    )

    # fix dtype for classification tasks
    # jenga relies on `categorical` dtype for this
    if _ID_TO_TASK_TYPE[task_id] == TaskType.BINARY or _ID_TO_TASK_TYPE[task_id] == TaskType.MULTI_CLASS:
        y = y.astype("category")

    # using `.squeeze("columns")` for `read_csv` returns `pd.Series` => ignoring mypy
    return X, y
