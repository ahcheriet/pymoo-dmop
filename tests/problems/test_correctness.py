import autograd.numpy as anp
import pytest

from pymoo.factory import get_problem
import os

from tests.util import path_to_test_resource
from dynamic_mop.cec2018 import *


problems = [
    ('DF1',[10,5]),('DF2',[10,5]),('DF3',[10,5]),('DF4',[10,5]),('DF5',[10,5]),('DF5', [10, 5]),
    ('DF6', [10, 5]), ('DF7', [10, 5]), ('DF8', [10, 5]), ('DF9', [10, 5]), ('DF10', [10, 5]),
    ('DF11', [10, 5]), ('DF12', [10, 5]), ('DF13', [10, 5])
]


@pytest.mark.parametrize('name,params', problems)
def test_problems(name, params):
    X, F, CV = load(name)

    if F is None:
        print("Warning: No correctness check for %s" % name)
        return

    problem = CEC2018(problemID=name, nt=params[0], taut=params[1])
    _F = problem.evaluate(X)

    if problem.n_obj == 1:
        F = F[:, None]

    assert anp.all(anp.abs(_F - F) < 0.00001)


def load(name, suffix=[]):
    path = path_to_test_resource("problems", *suffix)

    X = anp.loadtxt(os.path.join(path, "%s.x" % name))

    try:
        F = anp.loadtxt(os.path.join(path, "%s.f" % name))

        CV = None
        if os.path.exists(os.path.join(path, "%s.cv" % name)):
            CV = anp.loadtxt(os.path.join(path, "%s.cv" % name))

    except:
        return X, None, None

    return X, F, CV
