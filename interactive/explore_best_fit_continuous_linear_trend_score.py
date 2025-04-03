import numpy as np

# Clever idea, use least squares to fit a continuous piecewise linear function.
# Continuity is enforced by using the same intercept at the kink point.


# %%
# Clever idea, use least squares to fit a continuous piecewise linear function.
# Continuity is enforced by using the same intercept at the kink point.
def optimal_continuous_piecewise_linear_v2(signal, m):
    """
    Find the optimal continuous piecewise linear trend with a kink at index m.

    This function fits a continuous piecewise linear function with a single kink
    at index m that minimizes the squared error to the signal.

    Parameters
    ----------
    signal : np.ndarray
        1D array containing the signal
    m : int
        Index where the kink should be placed (0 < m < len(signal)-1)

    Returns
    -------
    fitted_trend : np.ndarray
        Fitted continuous piecewise linear trend values
    params : tuple
        (intercept1, slope1, slope2) parameters of the piecewise linear function
    error : float
        Sum of squared errors between the signal and fitted trend
    """
    n = len(signal)
    if m <= 0 or m >= n - 1:
        raise ValueError("m must be between 1 and len(signal)-2")

    # Create design matrix X for the piecewise linear regression
    # We need three parameters: intercept, slope1, and slope2
    X = np.zeros((n, 3))

    # First column is all ones (intercept)
    X[:, 0] = 1

    # Second column is the time index for the first segment
    X[:, 1] = np.arange(n)

    # Third column represents the additional slope for the second segment
    # For continuity at x=m:
    #     f(x) = intercept + slope1*m + slope2*max(0,x-m)
    # For x < m: f(x) = intercept + slope1*x
    # For x > m: f(x) = intercept + slope1*m + slope2*(x-m)
    # At x=m: intercept + slope1*m + slope2*max(0,x-m)
    # At x=m: intercept + slope1*m + slope2*max(0,m-m),
    #          = intercept + slope1*m + slope2*0,
    #          = intercept + slope1*m.
    # At x=m + 1: intercept + slope1*(m + 1) + slope2*max(0,x-m)
    ## NOT: Kontinuerlig i siste punktet av forrige intervall [start, split - 1],
    # i stedet for i første punktet av neste intervall [split, end - 1].

    # Kontinuerlig i første punktet av neste intervall [split, end - 1]:
    # X[m:, 2] = np.arange(n - m)

    # Kontinuerlig i siste punktet av forrige intervall [start, split - 1]:
    X[m:, 2] = np.arange(1, n - m + 1)

    # Solve for the parameters using least squares
    beta, residuals, rank, singular_values = np.linalg.lstsq(X, signal)

    # Extract parameters
    intercept = beta[0]
    slope1 = beta[1]
    slope2 = beta[2]  # Additional slope (total slope for segment 2 is slope1 + slope2)

    # Compute the fitted values
    # fitted_trend = np.zeros(n)
    # fitted_trend[: m + 1] = intercept + slope1 * np.arange(m + 1)
    # fitted_trend[m:] = (
    #     intercept + slope1 * m + slope1 * np.arange(n - m) + slope2 * np.arange(n - m)
    # )
    fitted_trend = X @ beta

    # Calculate error
    error = np.sum(np.square(signal - fitted_trend))
    # print("Residuals:", residuals)
    # print("Error:", error)
    # print("Rank:", rank)
    # print("Singular values:", singular_values)

    return fitted_trend, (intercept, slope1, slope1 + slope2), error


# %%
def continuous_piecewise_linear_trend_contrast_function(
    signal: np.ndarray,
    first_interval_inclusive_start: int,
    second_interval_inclusive_start: int,
    non_inclusive_end: int,
):
    # Assume 'start' is the first index of the data, perform inner product with the
    # desired segment of the data to get the cost.
    assert (
        first_interval_inclusive_start + 1
        < second_interval_inclusive_start
        < non_inclusive_end
    )
    # interval_length = non_inclusive_end - first_interval_inclusive_start
    # Translate named parameters to the NOT-paper sytax.
    # We are zero-indexing the data, whilst the paper is one-indexing.
    s = first_interval_inclusive_start - 1
    e = non_inclusive_end - 1
    b = second_interval_inclusive_start - 1
    l = e - s
    alpha = np.sqrt(
        6.0
        / (l * (l**2 - 1.0) * (1.0 + (e - b + 1.0) * (b - s) + (e - b) * (b - s - 1.0)))
    )
    beta = np.sqrt(((e - b + 1.0) * (e - b)) / ((b - s - 1.0) * (b - s)))

    first_interval_slope = 3.0 * (b - s) + (e - b) - 1.0
    first_interval_constant = b * (e - s - 1.0) + 2.0 * (s + 1.0) * (b - s)

    second_interval_slope = 3.0 * (e - b) + (b - s) + 1.0
    second_interval_constant = b * (e - s - 1.0) + 2.0 * e * (e - b + 1)

    signal_length = len(signal)
    contrast_vector = np.zeros(signal_length)
    for t in range(s + 1, b + 1):
        # t is the index of the data.
        contrast_vector[t] = (
            alpha * beta * (first_interval_slope * t - first_interval_constant)
        )

    for t in range(b + 1, e + 1):
        # t is the index of the data.
        contrast_vector[t] = (-alpha / beta) * (
            second_interval_slope * t - second_interval_constant
        )

    contrast_value = np.abs(np.dot(contrast_vector, signal))
    return contrast_value


def continuous_piecewise_linear_trend_squared_contrast(
    signal: np.ndarray,
    first_interval_inclusive_start: int,
    second_interval_inclusive_start: int,
    non_inclusive_end: int,
):
    # Assume 'start' is the first index of the data, perform inner product with the
    # desired segment of the data to get the cost.
    assert (
        first_interval_inclusive_start + 1
        < second_interval_inclusive_start
        < non_inclusive_end
    )
    # interval_length = non_inclusive_end - first_interval_inclusive_start
    # Translate named parameters to the NOT-paper sytax.
    # We are zero-indexing the data, whilst the paper is one-indexing.
    s = first_interval_inclusive_start - 1
    e = non_inclusive_end - 1
    b = second_interval_inclusive_start - 1
    l = e - s
    alpha = np.sqrt(
        6.0 / (l * (l**2 - 1) * (1 + (e - b + 1) * (b - s) + (e - b) * (b - s - 1)))
    )
    beta = np.sqrt(((e - b + 1) * (e - b)) / ((b - s - 1) * (b - s)))

    first_interval_slope = 3.0 * (b - s) + (e - b) - 1.0
    first_interval_constant = b * (e - s - 1.0) + 2.0 * (s + 1.0) * (b - s)

    second_interval_slope = 3.0 * (e - b) + (b - s) + 1.0
    second_interval_constant = b * (e - s - 1.0) + 2.0 * e * (e - b + 1)

    # Accumulate the contrast value inner product:
    contrast = 0.0
    for t in range(s + 1, b + 1):
        contrast += (
            alpha * beta * (first_interval_slope * t - first_interval_constant)
        ) * signal[t]

    for t in range(b + 1, e + 1):
        contrast += (
            (-alpha / beta) * (second_interval_slope * t - second_interval_constant)
        ) * signal[t]

    return np.square(contrast)


# %%
from skchange.change_scores import ContinuousLinearTrendScore

# Test the functions:
if __name__ == "__main__":
    # Example signal with a change point
    # signal = np.array([1, 2, 3, 4, 5, 7, 9, 11, 13, 15])
    # signal = np.linspace(start=1, stop=12, num=10)
    signal = np.linspace(start=1, stop=12, num=10) + np.random.normal(
        loc=0, scale=0.5, size=10
    )

    best_fit_linear_trend_score = ContinuousLinearTrendScore()
    best_fit_linear_trend_score.fit(signal)

    #  split_index = 4
    argmax_contrast_score = -1
    max_contrast_value = -np.inf

    argmax_trend_score = -1
    max_trend_score_value = -np.inf

    for split_index in range(2, len(signal) - 1):
        # Find the optimal continuous piecewise linear trend with a kink at index 5
        # Test the second version

        linreg_piecewise_trend_score = best_fit_linear_trend_score.evaluate(
            np.array([[0, split_index, len(signal)]])
        )
        print("skchange trend score:", linreg_piecewise_trend_score[0, 0])
        if linreg_piecewise_trend_score[0, 0] > max_trend_score_value:
            max_trend_score_value = linreg_piecewise_trend_score[0, 0]
            argmax_trend_score = split_index

        contrast_value = continuous_piecewise_linear_trend_contrast_function(
            signal,
            first_interval_inclusive_start=0,
            second_interval_inclusive_start=split_index,
            non_inclusive_end=len(signal),
        )
        print("Contrast value:", contrast_value)
        print("Squared Contrast value:", contrast_value**2)
        direct_squared_contrast_value = (
            continuous_piecewise_linear_trend_squared_contrast(
                signal,
                first_interval_inclusive_start=0,
                second_interval_inclusive_start=split_index,
                non_inclusive_end=len(signal),
            )
        )
        print("Direct squared contrast value:", direct_squared_contrast_value)

        if contrast_value > max_contrast_value:
            max_contrast_value = contrast_value
            argmax_contrast_score = split_index

        contrast_score_diff = linreg_piecewise_trend_score - contrast_value**2
        relative_diff = (contrast_score_diff / linreg_piecewise_trend_score) * 100
        print("Contrast score difference:", contrast_score_diff)
        print("Relative difference:", relative_diff)

    print("\nMax trend score value:", max_trend_score_value)
    print("Argmax trend score:", argmax_trend_score)

    print("Argmax contrast score:", argmax_contrast_score)
    print("Max contrast value:", max_contrast_value)
