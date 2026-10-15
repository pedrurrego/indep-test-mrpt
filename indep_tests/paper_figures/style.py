"""Publication matplotlib styling for the paper figures.

Defaults to vector output with embedded, editable fonts (``pdf.fonttype = 42``),
serif text and STIX math to sit cleanly next to the ``sn-jnl`` body text. Use as
a context manager so the global rcParams are never mutated for other callers::

    with paper_style():
        fig, ax = plt.subplots(...)
        ...
        fig.savefig("figure.pdf")
"""

from __future__ import annotations

import contextlib

import matplotlib as mpl

# Render headless: savefig still picks the pdf backend from the .pdf suffix.
mpl.use("Agg")

_PAPER_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.format": "pdf",
    # Embed fonts as editable TrueType — required/expected by many publishers.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.3,
    "lines.markersize": 4,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
}


@contextlib.contextmanager
def paper_style():
    """Apply the publication rcParams for the duration of the ``with`` block."""
    with mpl.rc_context(_PAPER_RC):
        yield
