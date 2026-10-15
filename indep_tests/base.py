from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TestResult:
    """Result of an independence test."""

    statistic: float
    p_value: float
    n: int
    B: int | None = None
    test_name: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


class IndependenceTest(ABC):
    """Abstract base class for independence tests.

    All configuration happens at __init__ time.
    The two core methods take only data (and bootstrap params for test()).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the test."""
        ...

    @abstractmethod
    def statistic(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute the raw test statistic.

        Parameters
        ----------
        x : np.ndarray, shape (n,)
        y : np.ndarray, shape (n,)

        Returns
        -------
        float
        """
        ...

    @abstractmethod
    def test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        B: int = 1000,
        seed: int | None = None,
    ) -> TestResult:
        """Run the full independence test with p-value computation.

        Parameters
        ----------
        x : np.ndarray, shape (n,)
        y : np.ndarray, shape (n,)
        B : int
            Number of bootstrap/permutation replicates (ignored by parametric tests).
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        TestResult
        """
        ...
