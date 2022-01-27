import numpy as np
import pytest

from pymoo.factory import get_problem
from tests.problems.test_correctness import load

PROBLEMS = ["DF1", "DF2", "DF3", "DF4", "DF5"]


@pytest.mark.parametrize('name', PROBLEMS)
def test_problems(name):
    problem = get_problem(name)

    X, F, CV = load(name.upper())
    _F, _CV, _G = problem.evaluate(X, return_values_of=["F", "CV", "G"])

    if _G.shape[1] > 1:
        # We need to do a special CV calculation to tests for correctness since
        # the original code does not sum the violations but takes the maximum
        _CV = np.max(_G, axis=1)[:, None]
        _CV = np.maximum(_CV, 0)

    np.testing.assert_allclose(_F, F)
    np.testing.assert_allclose(_CV[:, 0], CV)
test_problems('DF1')


