"""Functions for loading datasets that ship with `skchange`."""

import csv
import os
from datetime import datetime, timezone


def load_hvac_system_data(as_frame: bool = False) -> dict:
    """Load the heating, ventilation and air conditioning (HVAC) system dataset.

    The dataset contains time series of vibration magnitude measurements from two
    different HVAC systems. 30 days of sensor measurements are available for each unit,
    with a sampling rate of 10 minutes.

    The aim of analysing the data is to detect when each HVAC system normally
    turns on and off, such that anomalies from their regular schedule can be detected.
    True labels are not available, but it is fairly easy to identify the normal
    schedule from the data and observe when the units deviate from it.

    The data has been provided by the company Soundsensing:
    https://www.soundsensing.no/.

    Parameters
    ----------
    as_frame : bool, default=False
        If True, the data is returned as a ``pandas.DataFrame``. Requires pandas to
        be installed. If False (default), the data is returned as a dict of numpy
        arrays.

    Returns
    -------
    data : dict
        Dictionary with the following keys:

        * ``"data"`` : ndarray of shape (n_samples, 1) of float
            The vibration magnitude measurements.
        * ``"feature_names"`` : list[str]
            ``["vibration"]``.
        * ``"time"`` : ndarray of shape (n_samples,) of ``datetime64[ns]``
            Timestamp of each measurement.
        * ``"unit_id"`` : ndarray of shape (n_samples,) of int
            Identifier of the HVAC unit each measurement belongs to.
        * ``"description"`` : str
            Description of the dataset.

        If ``as_frame=True``, the dict also contains:

        * ``"frame"`` : pandas.DataFrame
            DataFrame with a ``MultiIndex`` of ``("unit_id", "time")`` and a single
            ``"vibration"`` column.
    """
    import numpy as np

    this_file_dir = os.path.dirname(__file__)
    file_path = os.path.join(this_file_dir, "data", "hvac_system", "data.csv")

    times: list[datetime] = []
    vibrations: list[float] = []
    unit_ids: list[int] = []
    with open(file_path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # header: "", "time", "vibration", "unit_id"
        for row in reader:
            _, t, v, u = row
            times.append(datetime.fromisoformat(t).astimezone(timezone.utc))
            vibrations.append(float(v))
            unit_ids.append(int(u))

    time_arr = np.array([t.replace(tzinfo=None) for t in times], dtype="datetime64[ns]")
    vibration_arr = np.asarray(vibrations, dtype=np.float64).reshape(-1, 1)
    unit_id_arr = np.asarray(unit_ids, dtype=np.int64)

    descr = (
        "HVAC system vibration time series from Soundsensing"
        " (https://www.soundsensing.no/). "
        "30 days of 10-minute sampled vibration measurements from two HVAC units."
    )

    out = {
        "data": vibration_arr,
        "feature_names": ["vibration"],
        "time": time_arr,
        "unit_id": unit_id_arr,
        "description": descr,
    }

    if as_frame:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "load_hvac_system_data(as_frame=True) requires pandas to be installed."
            ) from exc

        frame = pd.DataFrame(
            {"vibration": vibration_arr.ravel()},
            index=pd.MultiIndex.from_arrays(
                [unit_id_arr, pd.to_datetime(time_arr)],
                names=["unit_id", "time"],
            ),
        )
        out["frame"] = frame

    return out
