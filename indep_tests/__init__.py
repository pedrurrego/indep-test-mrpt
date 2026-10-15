"""Independence tests unified interface."""

from indep_tests.base import IndependenceTest, TestResult
from indep_tests.registry import get_test_class, test_names

__all__ = ["IndependenceTest", "TestResult", "get_test_class", "test_names"]
