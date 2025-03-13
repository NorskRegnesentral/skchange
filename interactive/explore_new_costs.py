import numpy as np

from skchange.costs._linear_trend_cost import fit_linear_trend

ts = np.array([1, 2, 3, 4, 5])
xs = 4 * ts + 1
slope, intercept = fit_linear_trend(ts, xs)
print(slope, intercept)
reconstructed_xs = slope * ts + intercept
