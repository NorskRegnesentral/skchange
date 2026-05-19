"""Deprecation utilities for the old skchange API.

The current pandas/sktime-based API is being replaced by the new
sklearn-compatible API previewed under :mod:`skchange.new_api`. See the
migration guide for details:

https://github.com/NorskRegnesentral/skchange/blob/main/skchange/new_api/MIGRATION_GUIDE.md
"""

import warnings

_MIGRATION_GUIDE_URL = (
    "https://github.com/NorskRegnesentral/skchange/blob/main/"
    "skchange/new_api/MIGRATION_GUIDE.md"
)

_OLD_API_MSG = (
    "The current skchange API will be removed in 0.17.0 and replaced by the "
    "API currently previewed in `skchange.new_api`. To keep using the current "
    "API, pin `skchange<0.17`. See the migration guide for details: "
    f"{_MIGRATION_GUIDE_URL}"
)

_warned = False


def warn_old_api() -> None:
    """Emit a one-shot ``FutureWarning`` about the old-API removal in 0.17.0."""
    global _warned
    if _warned:
        return
    _warned = True
    warnings.warn(_OLD_API_MSG, FutureWarning, stacklevel=3)
