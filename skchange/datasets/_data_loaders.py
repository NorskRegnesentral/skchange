"""Functions for loading datasets that ships with `skchange`."""

import os

import pandas as pd


def load_air_handling_units():
    """Load the air handling units dataset.

    The dataset contains time series of vibration magnitude measurements from two
    different air handling units, each monitored over a period of 30 days. The data is
    sampled every 10 minutes.

    The aim of analysing the data is to detect when the air handling units normally
    turn on and off, such that anomalies from their regular schedule can be detected.

    The data has been provided by Soundsensing AS.

    Returns
    -------
    pd.DataFrame
        The air handling unit dataset. It has a multi-index with two levels:

        1. "unit_id": A string identifier for each air handling unit.
        2. "time": A datetime index with the time of each measurement.

        There's one column:

        1. "vibration": A float column with the vibration magnitude measurements.
    """
    this_file_dir = os.path.dirname(__file__)
    file_path = os.path.join(this_file_dir, "data", "air_handling_unit", "data.csv")
    df = pd.read_csv(file_path).iloc[:, 1:]
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index(["unit_id", "time"])
    return df
