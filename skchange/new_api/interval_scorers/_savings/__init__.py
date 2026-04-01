"""A range of savings implemented as interval scorers.

Savings are test statistics that measure the improvement in cost/loss when using an
optimal/estimated vs. a fixed/baseline segment parameter value:

```
X_sub = X[start:end]
saving(X_sub) = cost(X_sub, fixed_param) - cost(X_sub, optimal_param)
```

Large savings for an interval indicate that the baseline parameter is a poor fit for
the data in that interval, and a signal for `start` and `end` being changepoints.

Imports should be from skchange.new_api.interval_scorers, not from this submodule.
"""
