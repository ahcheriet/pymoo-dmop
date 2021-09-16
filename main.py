import autograd.numpy as anp
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt
#from pymoo.performance_indicator.igd import IGD
from dproblems import *
from CEC2018 import *
from pymoo.factory import get_performance_indicator

# nt: severity of change
# taut: frequency of change
# tauT: maximum number of generation
# tau : current generation
# examples of nt and taut values
nt_ = 5
taut_ = 10
tauT = 200

# TODO use an abstract class for all dynamic problems

class DynamicMOP(Problem):
    """
    abstract class for a dynamic multi-objective problem

    """

    def __init__(self, n_var, n_obj, nt=nt_, taut=taut_):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0, xu=1)
        self.nt = nt
        self.taut = taut
        self.tau = 0 # current iteration

    def get_current_t(self):
        t = 1 / self.nt
        t = t * floor(self.tau / self.taut)
        return t

    def _calc_pareto_front(self, n_pareto_points=100):
        pass

    def get_pf_t(self):
        return self._calc_pareto_front()


class DMOP2(ElementwiseProblem):
    """DMOP2 dynamic benchmark problem
    """
    def __init__(self, nt=nt_, taut=taut_):
        super().__init__(n_var=10,
                         n_obj=2,
                         xl=0,
                         xu=1)
        self.nt = nt
        self.taut = taut
        self.tau = 0 # current iteration

    def get_current_t(self, n_gen):
        self.tau = n_gen
        t = 1 / self.nt
        t = t * floor(self.tau / self.taut)
        return t

    def _evaluate(self, X, out, *args, algorithm=None, **kwargs):
        self.t = self.get_current_t(algorithm.n_gen)
        f1, f2 = dMOP2(X, self.t)
        out["F"] = [f1, f2]

    def _calc_pareto_front(self, n_pareto_points=100):
        H = 0.75 * sin(0.5 * pi * self.t) + 1.25
        x = anp.linspace(0, 1, n_pareto_points)
        pf = anp.column_stack([x, 1-pow(x,H)])
        return pf

    def get_pf_t(self):
        return self._calc_pareto_front()


class DIMP2b(Problem):
    """DIMP2 dynamic benchmark problem
    """
    def __init__(self, nt=nt_, taut=taut_):
        super().__init__(n_var=10,
                         n_obj=2,
                         xl=[0]+[-2]*9,
                         xu=[1]+[2]*9,
                         elementwise_evaluation=True)
        self.nt = nt
        self.taut = taut
        self.tau = 0 # current iteration

    def get_current_t(self, n_gen):
        self.tau = n_gen
        t = 1 / self.nt
        t = t * floor(self.tau / self.taut)
        return t

    def _evaluate(self, X, out, *args, algorithm=None, **kwargs):
        self.t = self.get_current_t(algorithm.n_gen)
        f1, f2 = DIMP2(X, self.t)
        out["F"] = [f1, f2]

    def _calc_pareto_front(self, n_pareto_points=100):
        x = anp.linspace(0, 1, n_pareto_points)
        pf = anp.column_stack([x, 1-np.sqrt(x)])
        return pf

    def get_pf_t(self):
        return self._calc_pareto_front()

class HE2b(Problem):
    """HE2 dynamic benchmark problem
    """
    def __init__(self, nt=nt_, taut=taut_):
        super().__init__(n_var=30,
                         n_obj=2,
                         xl=0,
                         xu=1,
                         elementwise_evaluation=True)
        self.nt = nt
        self.taut = taut
        self.tau = 0 # current iteration

    def get_current_t(self, n_gen):
        self.tau = n_gen
        t = 1 / self.nt
        t = t * floor(self.tau / self.taut)
        return t

    def _evaluate(self, X, out, *args, algorithm=None, **kwargs):
        self.t = self.get_current_t(algorithm.n_gen)
        f1, f2 = HE2(X, self.t)
        out["F"] = [f1, f2]

    def _calc_pareto_front(self, n_pareto_points=100):
        H = 0.75 * sin(0.5 * pi * self.t) + 1.25  # time may be incorrect
        x = anp.linspace(0, 1, n_pareto_points)
        f2 = (1-np.power(np.sqrt(x), H))-(np.power(x, H)*np.sin(0.5*pi*x))
        pf = anp.column_stack([x, f2])
        return pf

    def get_pf_t(self):
        return self._calc_pareto_front()


class HE7b(Problem):
    """HE7 dynamic benchmark problem
    """
    def __init__(self, nt=nt_, taut=taut_):
        super().__init__(n_var=10,
                         n_obj=2,
                         xl=[0,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                         xu=[1,1,1,1,1,1,1,1,1,1],
                         elementwise_evaluation=True)
        self.nt = nt
        self.taut = taut
        self.tau = 0 # current iteration

    def get_current_t(self, n_gen):
        self.tau = n_gen
        t = 1 / self.nt
        t = t * floor(self.tau / self.taut)
        return t

    def _evaluate(self, X, out, *args, algorithm=None, **kwargs):
        self.t = self.get_current_t(algorithm.n_gen)
        f1, f2 = HE7(X, self.t)
        out["F"] = [f1, f2]

    def _calc_pareto_front(self, n_pareto_points=100):
        H = 0.75*sin(0.5*pi*self.t)+1.25  # current time may be incorrect
        x = anp.linspace(0, 1, n_pareto_points)
        f2 = (2-np.sqrt(x))*(1-np.power(x/(2-np.sqrt(x)),H))
        pf = anp.column_stack([x, f2])
        return pf

    def get_pf_t(self):
        return self._calc_pareto_front()

class HE9b(Problem):
    """HE9 dynamic benchmark problem
    """
    def __init__(self, nt=nt_, taut=taut_):
        super().__init__(n_var=10,
                         n_obj=2,
                         xl=0,
                         xu=1,
                         elementwise_evaluation=True)
        self.nt = nt
        self.taut = taut
        self.tau = 0 # current iteration

    def get_current_t(self, n_gen):
        self.tau = n_gen
        t = 1 / self.nt
        t = t * floor(self.tau / self.taut)
        return t

    def _evaluate(self, X, out, *args, algorithm=None, **kwargs):
        self.t = self.get_current_t(algorithm.n_gen)
        f1, f2 = HE9(X, self.t)
        out["F"] = [f1, f2]

    def _calc_pareto_front(self, n_pareto_points=100):
        H = 0.75 * sin(0.5 * pi * self.t) + 1.25 # current time may be incorrect
        x = anp.linspace(0, 1, n_pareto_points)
        f2 = (2-np.sqrt(x))*(1-np.power(x/(2-np.sqrt(x)),H))
        pf = anp.column_stack([x, f2])
        return pf

    def get_pf_t(self):
        return self._calc_pareto_front()

class CEC2018(Problem):
    """HE9 dynamic benchmark problem
    """
    def __init__(self, nt=nt_, taut=taut_, problemID='DF1'):
        super().__init__(n_var=10,
                         n_obj=2,
                         xl=0,
                         xu=1,
                         elementwise_evaluation=True)
        self.problemID = problemID
        self.nt = nt
        self.taut = taut
        self.tau = 0 # current iteration

    def _evaluate(self, X, out, *args, **kwargs):
        f = cec2018_DF(problemID=self.problemID, x=X, tau=self.tau, nt=self.nt, taut=self.taut)
        out["F"] = list(f)

    def get_pf_t(self):
        return self._calc_pareto_front()


algorithm = NSGA2(pop_size=200)
problem = DMOP2()

PF = []


# This call back is to store the result(pf) in each time the problem changes
def callback(ex_algorithm):
    if (ex_algorithm.problem.tau+1) % ex_algorithm.problem.taut == 0:
        PF.append(ex_algorithm.problem.get_pf_t())  # pareto_front()


algorithm.callback = callback


res = minimize(problem,
               algorithm,
               ("n_gen", tauT),
               save_history=True,
               verbose=False,
               seed=np.random.randint(1,1000))

n_gen = []    # corresponding number of function evaluations\
F = []          # the objective space values in each generation
cv = []         # constraint violation in each generation


for algorithm in res.history:
    n_gen.append(algorithm.n_gen)
    opt = algorithm.opt
    cv.append(opt.get("CV").min())
    feas = np.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feas]
    F.append(_F)

igd = []
# this is to calculate the IGD every time the problem changes.

for idx, pf in enumerate(PF):
    normalize = False
    metric = get_performance_indicator("igd", pf)
    #IGD(pf=pf, normalize=normalize)
    igd = igd + [metric.do(f) for f in F[idx*taut_:(idx*taut_+taut_)]]

print(np.mean(igd))
plt.plot(n_gen, igd, '-o', markersize=4, linewidth=2, color="green")
plt.yscale("log")          # enable log scale if desired
plt.title("Convergence")
plt.xlabel("Iteration")
plt.ylabel("IGD")
plt.show()

# just plotting the PFs
plot = Scatter()
for pf in PF:
    plot.add(pf, plot_type="line", color="black", linewidth=2)
plot.show()