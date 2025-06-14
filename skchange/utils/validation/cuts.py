"""Utility functions for interval scorers."""

import numpy as np


def check_cuts_array(
    cuts: np.ndarray,
    min_size: int | None = None,
    last_dim_size: int = 2,
) -> np.ndarray:
    """Check array type cuts.

    Parameters
    ----------
    cuts : np.ndarray
        Array of cuts to check.
    min_size : int, optional (default=1)
        Minimum size of the intervals obtained by the cuts.
    last_dim_size : int, optional (default=2)
        Size of the last dimension.

    Returns
    -------
    cuts : np.ndarray
        The unmodified input cuts array.

    Raises
    ------
    ValueError
        If the cuts does not meet the requirements.
    """
    if min_size is None:
        min_size = 1

    if cuts.ndim != 2:
        raise ValueError("The cuts must be a 2D array.")

    if not np.issubdtype(cuts.dtype, np.integer):
        raise ValueError("The cuts must be of integer type.")

    if cuts.shape[-1] != last_dim_size:
        raise ValueError(
            "The cuts must be specified as an array with length "
            f"{last_dim_size} in the last dimension."
        )

    interval_sizes = np.diff(cuts, axis=1)
    if not np.all(interval_sizes >= min_size):
        raise ValueError(
            "All rows in `cuts` must be strictly increasing and each entry must"
            f" be more than min_size={min_size} apart."
        )
    return cuts
