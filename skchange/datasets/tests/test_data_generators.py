import pandas as pd
import pytest
from scipy.stats import (
    bernoulli,
    binom,
    expon,
    hypergeom,
    multivariate_normal,
    multivariate_t,
    norm,
    poisson,
    rv_continuous,
    rv_discrete,
    uniform,
)

from skchange.datasets import generate_changing_data, generate_piecewise_data

SCIPY_DISTRIBUTIONS = [
    norm(),
    uniform(),
    poisson(5),
    binom(n=10, p=0.5),
    expon(),
    multivariate_normal(),
    multivariate_t(),
    bernoulli(p=0.5),
    hypergeom(M=20, n=7, N=12),
]


@pytest.mark.parametrize("distribution", SCIPY_DISTRIBUTIONS)
def test_generate_piecewise_data(distribution: rv_continuous | rv_discrete):
    length = 10
    df = generate_piecewise_data(
        distributions=[distribution],
        lengths=[length],
        random_state=42,
    )
    assert df.shape[0] == length
    assert df.columns == pd.RangeIndex(start=0, stop=df.shape[1])


def test_generate_piecewise_data_invalid_distributions():
    with pytest.raises(ValueError):
        generate_piecewise_data(
            distributions=[norm],
            lengths=[100, 50],
            random_state=42,
        )


def test_generate_piecewise_data_invalid_lengths():
    with pytest.raises(ValueError):
        generate_piecewise_data(
            distributions=[norm],
            lengths=[-1, 2],  # Mismatched lengths
        )


def test_generate_piecewise_data_random_state():
    length = 100
    df1 = generate_piecewise_data(
        distributions=[norm],
        lengths=length,
        random_state=42,
    )
    df2 = generate_piecewise_data(
        distributions=[norm],
        lengths=length,
        random_state=42,
    )
    pd.testing.assert_frame_equal(df1, df2)

    df3 = generate_piecewise_data(
        distributions=[norm],
        lengths=[length],
    )

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df1, df3)


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
    with pytest.raises(ValueError):
        generate_changing_data(
            n=100,
            changepoints=110,  # Invalid changepoint.
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
