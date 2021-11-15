from dynamic_mop.CEC2018 import *
from dynamic_mop.DynamicMOP import  DynamicMOP


class CEC2018(DynamicMOP):
    """HE9 dynamic benchmark problem
    """
    def __init__(self, nt=10, taut=200, problemID='DF1'):
        super().__init__(n_var=10,
                         n_obj=2,
                         xl=0,
                         xu=1)
        self.problemID = problemID
        self.nt = nt
        self.taut = taut
        self.tau = 0 # current iteration
        self.xl, self.xu = get_bounds(problem_id=problemID, n_vars=10) # n_vars = 10 from the CEC2018 Competition

    def _calc_pareto_front(self, n_pareto_points=100):
        self.t = self.get_current_t()
        pf = cec2018_DF_PF(probID=self.problemID, t=self.t, n_points=n_pareto_points)
        return pf

    def _evaluate(self, X, out, *args, **kwargs):
        self.t = self.get_current_t()
        f = cec2018_DF(problemID=self.problemID, x=X, t=self.t)
        out["F"] = list(f.values())

    def get_pf_t(self):
        return self._calc_pareto_front()

