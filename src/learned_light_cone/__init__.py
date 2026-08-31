"""Reusable diagnostics and operators for learned light-cone analysis."""

from .lightcone import (
    cone_leakage,
    cone_report,
    examples,
    localized_bump,
    perturbation_response,
    tail_exponent,
    torch_forward,
)

__all__ = [
    "__version__",
    "localized_bump",
    "perturbation_response",
    "cone_leakage",
    "tail_exponent",
    "cone_report",
    "torch_forward",
    "examples",
]
__version__ = "0.1.0"
