import numpy as np

from skchange.new_api.datasets import load_hvac_system_data


def test_load_hvac_system_data():
    data = load_hvac_system_data()

    # Default (numpy) interface
    assert isinstance(data, dict)
    assert set(data.keys()) >= {
        "data",
        "feature_names",
        "time",
        "unit_id",
        "description",
    }
    assert isinstance(data["data"], np.ndarray)
    assert data["data"].ndim == 2
    assert data["data"].shape[1] == 1
    assert data["feature_names"] == ["vibration"]
    assert data["data"].dtype.kind == "f"
    assert data["time"].dtype.kind == "M"
    assert data["unit_id"].dtype.kind in ("i", "u")


def test_load_hvac_system_data_as_frame():
    import pandas as pd

    data = load_hvac_system_data(as_frame=True)
    frame = data["frame"]
    assert isinstance(frame, pd.DataFrame)
    assert isinstance(frame.index, pd.MultiIndex)
    assert frame.index.names == ["unit_id", "time"]
    assert list(frame.columns) == ["vibration"]
    assert pd.api.types.is_float_dtype(frame["vibration"])
