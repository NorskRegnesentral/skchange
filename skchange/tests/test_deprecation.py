"""Tests for the old-API deprecation warning."""

import importlib
import subprocess
import sys
import warnings

import pytest

OLD_API_SUBPACKAGES = [
    "skchange.anomaly_detectors",
    "skchange.anomaly_scores",
    "skchange.change_detectors",
    "skchange.change_scores",
    "skchange.compose",
    "skchange.costs",
    "skchange.penalties",
]


def _reimport_fresh(module_name: str) -> None:
    """Import ``module_name`` in a fresh subprocess so the one-shot flag resets."""
    code = f"import warnings\nwarnings.simplefilter('always')\nimport {module_name}\n"
    # S603: inputs are sys.executable and module names from a fixed allowlist.
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-W", "always", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stderr


@pytest.mark.parametrize("module_name", OLD_API_SUBPACKAGES)
def test_old_api_subpackage_emits_future_warning(module_name):
    """Importing any old-API subpackage emits a FutureWarning."""
    stderr = _reimport_fresh(module_name)
    assert "FutureWarning" in stderr, stderr
    assert "0.17.0" in stderr, stderr
    assert "MIGRATION_GUIDE.md" in stderr, stderr


def test_new_api_does_not_emit_future_warning():
    """Importing ``skchange.new_api`` must not trigger the old-API warning."""
    stderr = _reimport_fresh("skchange.new_api")
    assert "FutureWarning" not in stderr, stderr


def test_warn_old_api_fires_once_per_process():
    """``warn_old_api`` is one-shot within a single interpreter."""
    # Reset the one-shot flag so this test is independent of import order.
    deprecation = importlib.import_module("skchange._deprecation")
    deprecation._warned = False

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        deprecation.warn_old_api()
        deprecation.warn_old_api()

    future_warnings = [w for w in captured if issubclass(w.category, FutureWarning)]
    assert len(future_warnings) == 1
    assert "0.17.0" in str(future_warnings[0].message)
    assert "MIGRATION_GUIDE.md" in str(future_warnings[0].message)
