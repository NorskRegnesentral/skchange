"""Centralised typing definitions for the new skchange API."""

from __future__ import annotations

try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python 3.8-3.10

import numpy as np

# numpy's ArrayLike definition can in some cases cause issues with type checkers,
# but we find it more convenient than using Any. Defining it here allows us to easily
# modify it in the future if needed, and provides a single source of truth for the
# expected array-like types across the new API.
ArrayLike = np.typing.ArrayLike

__all__ = [
    "ArrayLike",
    "Self",
]
