"""Penalties for change and anomaly detection."""

from skchange.new_api.penalties._constant import bic_penalty, chi2_penalty
from skchange.new_api.penalties._linear import linear_chi2_penalty, linear_penalty
from skchange.new_api.penalties._nonlinear import mvcapa_penalty, nonlinear_chi2_penalty

__all__ = [
    "bic_penalty",
    "chi2_penalty",
    "linear_penalty",
    "linear_chi2_penalty",
    "nonlinear_chi2_penalty",
    "mvcapa_penalty",
]
