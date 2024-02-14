from random import seed as random_seed
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def set_seed(seed: Optional[int]) -> None:
    if seed:
        random_seed(seed)
        np.random.seed(seed)


def is_categorical(column: ArrayLike, n_samples: int = 1000, max_unique_fraction: float = 0.2) -> bool:
    """Check if `column` type is categorical.

    A heuristic to check whether a `column` is categorical:
    a column is considered categorical (as opposed to a plain text column)
    if the relative cardinality is `max_unique_fraction` or less.
    Thanks to:
        https://github.com/awslabs/datawig/blob/f641342d05e95485ed88503d3efd9c3cca3eb7ab/datawig/simple_imputer.py#L147

    Args:
        column (ArrayLike): pandas `Series` containing strings
        n_samples (int, optional): number of samples used for heuristic. Defaults to 1000.
        max_unique_fraction (float, optional): maximum relative cardinality. Defaults to 0.2.

    Returns:
        bool: `True` if the column is categorical according to the heuristic.
    """
    column = np.array(column)
    replace = len(column) < n_samples
    sample = np.random.choice(column, n_samples, replace=replace)
    unique_samples = pd.unique(sample)

    return (unique_samples.shape[0] / n_samples) <= max_unique_fraction
