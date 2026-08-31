"""Shared deterministic visual settings for canonical replay figures."""

from __future__ import annotations

from typing import Any

FIGURE_METADATA = {
    "Creator": "learned_light_cone",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}

COLORS = {
    "global": "#3b5b92",
    "local": "#c44e52",
    "control": "#7f7f7f",
    "accent": "#4c9f70",
}


def apply_style(matplotlib_module: Any) -> None:
    """Apply the compact style used by all canonical producers."""
    matplotlib_module.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "pdf.compression": 9,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
