from pymoo.core.problem import Problem, ElementwiseProblem
from dynamic_mop.functions import *
# TODO use an abstract class for all dynamic problems


class DynamicMOP(ElementwiseProblem):
    """
    abstract class for a dynamic multi-objective problem

    """

    def __init__(self, n_var, n_obj, nt=5, taut=10, xl=0, xu=1):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        self.nt = nt
        self.taut = taut
        self.tau = 0 # current iteration
        self.has_changed = False

    def _evaluate(self, x, out, *args, **kwargs):
        pass

    def get_current_t(self):
#        self.tau = t
        t = 1 / self.nt
        t = t * np.floor(self.tau / self.taut)
        return t

    def _calc_pareto_front(self, n_pareto_points=100):
        pass

    def get_pf_t(self):
        return self._calc_pareto_front()

