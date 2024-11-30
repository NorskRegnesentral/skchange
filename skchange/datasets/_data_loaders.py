"""Functions for loading datasets that ships with `skchange`."""

import os

import pandas as pd


def load_air_handling_unit():
    """Load the air handling unit dataset.

    The dataset contains a time series of vibration magnitude measurements from an
    air handling unit monitored over a period of 30 days. The data is sampled every
    10 minutes.

    The data has been provided by Soundsensing AS.

    Returns
    -------
    pd.DataFrame
        The air handling unit dataset. The index is time, and there is one column with
        the vibration magnitude measurements.
    """
    this_file_dir = os.path.dirname(__file__)
    file_path = os.path.join(
        this_file_dir, "data", "air_handling_unit", "air_handling_unit.csv"
    )
    df = pd.read_csv(file_path)[1:].set_index("time")
    df.index = pd.to_datetime(df.index)
    return df
