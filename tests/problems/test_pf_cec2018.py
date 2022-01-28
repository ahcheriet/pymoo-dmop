import autograd.numpy as anp
import pytest
from dynamic_mop.cec2018 import *
from tests.problems.test_correctness import load


problems=[
    ('DF1',[10,5]),('DF2',[10,5]),('DF3',[10,5]),('DF4',[10,5]),('DF5',[10,5]),('DF5', [10, 5]),
    ('DF6', [10, 5]), ('DF7', [10, 5]), ('DF8', [10, 5]), ('DF8', [10, 5]), ('DF10', [10, 5]),
    ('DF11', [10, 5]), ('DF12', [10, 5]), ('DF13', [10, 5]),('DF14', [10, 5])
]


@pytest.mark.parametrize('name,params', problems)
def test_problems(name, params):
    _, _, PF = load(name)

    if PF is None:
        print("Warning: No correctness check for %s" % name)
        return

    problem = CEC2018(problemID=name, nt=params[0], taut=params[1])
    _PF = problem._calc_pareto_front(n_pareto_points=1500)

    if problem.n_obj == 1:
        PF = PF[:, None]

    assert anp.all(anp.abs(_PF - PF) < 0.00001)
