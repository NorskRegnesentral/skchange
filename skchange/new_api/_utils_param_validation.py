"""Parameter validation utilities for sklearn compatibility.

This module provides a stable interface to sklearn's parameter validation system,
which relies on some private APIs. By centralizing these imports here, we:

1. Clearly document our dependency on sklearn's internal validation system
2. Make it easier to adapt if sklearn changes these private APIs
3. Provide a single place to add fallbacks or compatibility shims if needed

Requirements
------------
- scikit-learn >= 1.2 (for _fit_context and _param_validation module)

Notes
-----
These are considered private APIs by sklearn and may change between versions.
However, they are the recommended way to implement parameter validation
matching sklearn's own estimators, as documented in sklearn's developer guide.

If sklearn makes breaking changes to these APIs, we can add compatibility shims
in this module rather than updating imports throughout the codebase.
"""

from sklearn.base import _fit_context
from sklearn.utils._param_validation import (
    HasMethods,
    Interval,
    StrOptions,
)

__all__ = [
    "_fit_context",
    "HasMethods",
    "Interval",
    "StrOptions",
]
