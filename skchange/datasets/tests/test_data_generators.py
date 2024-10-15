import pytest

from skchange.datasets.generate import generate_changing_data


def test_generate_changing_data():
    n = 100
    generate_changing_data(
        n=n,
        changepoints=[25, 50],
        means=[3.0, 0.0, 5.0],
        variances=[1.0, 2.0, 3.0],
    )

    generate_changing_data(
        n,
        changepoints=40,
    )

    generate_changing_data(
        n=n,
        changepoints=[],
        means=1.0,
        variances=5.0,
    )


def test_generate_changing_data_with_multiple_changepoints():
    n = 100
    changepoints = [10, 30]  # 3 segements
    means = [1.0, 3.0]  # 2 means -> 3 is required.
    variances = [1.0, 3.0]  # 2 variances -> 3 is required.
    random_state = 1
    with pytest.raises(ValueError):
        generate_changing_data(
            n=n,
            changepoints=changepoints,
            means=means,
            variances=variances,
            random_state=random_state,
        )


def test_generate_changing_data_invalid_changepoints():
    n = 100
    changepoints = [10, 20, 110]  # Invalid changepoint
    means = [1.0, 2.0, 3.0]
    variances = [1.0, 2.0, 3.0]
    random_state = 1
    with pytest.raises(ValueError):
        generate_changing_data(
            n=n,
            changepoints=changepoints,
            means=means,
            variances=variances,
            random_state=random_state,
        )


def test_generate_changing_data_mismatched_lengths():
    n = 100
    changepoints = [10, 20]
    means = [1.0, 2.0, 3.0]
    variances = [1.0, 2.0]  # Mismatched length
    random_state = 1
    with pytest.raises(ValueError):
        generate_changing_data(
            n=n,
            changepoints=changepoints,
            means=means,
            variances=variances,
            random_state=random_state,
        )
