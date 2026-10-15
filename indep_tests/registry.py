"""Name -> test-class registry with lazy imports.

Single source of truth for the test names accepted by the CLIs (``run``,
``salahub_noise``, ``salahub_rbex``, the Makefile's ``list-tests``): MRPT
(``pug-grid``, its chi-squared calibration ``pug-grid-chi2`` and the
cell-shape variant ``pug-partition``) and the seven comparators of the paper.
"""

from __future__ import annotations

import importlib
import importlib.util

# name -> (module, class)
_REGISTRY: dict[str, tuple[str, str]] = {
    "qad": ("indep_tests.qad", "QADTest"),
    "pug-grid": ("indep_tests.pug_grid", "PUGGridTest"),
    "pug-grid-chi2": (
        "indep_tests.pug_grid_pearson_chi_squared",
        "PUGGridPearsonChiSquaredTest",
    ),
    "pug-partition": ("indep_tests.shape_ablation", "PUGPartitionTest"),
    "dcov": ("indep_tests.dcov", "DCovTest"),
    "salahub-binning": ("indep_tests.salahub_binning", "SalahubBinningTest"),
    "ma-mao-fes": ("indep_tests.ma_mao_fes", "MaMaoFESTest"),
    "zhang-bet": ("indep_tests.zhang_bet", "ZhangBETTest"),
    "heller-bins": ("indep_tests.heller_bins", "HellerBinsTest"),
    "cmiel-ledwina": ("indep_tests.cmiel_ledwina_test", "CmielLedwinaTest"),
}


def _available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def test_names() -> list[str]:
    """Registered test names whose implementation is present in this checkout."""
    return [name for name, (module, _) in _REGISTRY.items() if _available(module)]


def get_test_class(name: str):
    """Import and return the ``IndependenceTest`` subclass registered as ``name``."""
    try:
        module, cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown test: {name!r} (available: {', '.join(test_names())})"
        ) from None
    return getattr(importlib.import_module(module), cls)
