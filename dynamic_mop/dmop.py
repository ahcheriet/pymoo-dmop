import autograd.numpy as anp
import numpy as np
from pymoo.core.problem import Problem, ElementwiseProblem
from dynamic_mop.functions import *
from dynamic_mop.CEC2018 import *
from dynamic_mop.DynamicMOP import  DynamicMOP


class FDA2_deb(DynamicMOP):
    """DMOP2 dynamic benchmark problem
    """
    def __init__(self,nt =10, taut= 200):
        super().__init__(n_var=13,
                         n_obj=2,
                         nt=nt,
                         taut=taut)
        self.xl = -1.0 * np.ones(self.n_var)
        self.xl[0] = 0.0
        self.xu = 1.0 * np.ones(self.n_var)
        self.xu[0] = 1.0

        self.T_max = 200

    def get_current_t(self, t):
        self.tau = t
        tt = 2 * floor(self.tau / self.taut) * (self.taut/(self.T_max-self.taut))
        return tt

    def _evaluate(self, X, out, *args, t=1, **kwargs):
        self.t = self.get_current_t(t)
        f1, f2 = fda2_deb(X, self.t)
        out["F"] = [f1, f2]

    def _calc_pareto_front(self, n_pareto_points=100):
        H = 0.75 + 0.75 * np.sin(0.5*np.pi*self.t)
        H = 2 * np.sin(0.5 * np.pi * (self.t - 1))
        x = np.linspace(0.01, 1, n_pareto_points)
        # TODO Solve the inf result when x[0] = 0 and H(t)<0
        print(H, x[0], (1-np.power(x, (1/H)))[0], self.t)
        pf = anp.column_stack([x, 1-np.power(x, (H))])
        return pf


class DMOP2(DynamicMOP):
    """DMOP2 dynamic benchmark problem
    """
    def __init__(self,nt =10, taut= 200):
        super().__init__(n_var=10,
                         n_obj=2,
                         nt=nt,
                         taut=taut)

    def get_current_t(self):
        #self.tau = t
        t = 1 / self.nt
        t = t * floor(self.tau / self.taut)
        return t

    def _evaluate(self, X, out, *args, t=1, **kwargs):
        self.t = self.get_current_t()
        f1, f2 = dMOP2(X, self.t)
        out["F"] = [f1, f2]

    def _calc_pareto_front(self, n_pareto_points=100):
        H = 0.75 * np.sin(0.5 * np.pi * self.t) + 1.25
        x = np.linspace(0, 1, n_pareto_points)
        pf = anp.column_stack([x, 1-pow(x, H)])
        return pf


class DIMP2b(DynamicMOP):
    """DIMP2 dynamic benchmark problem
    """
    def __init__(self, nt=10, taut=200):
        super().__init__(n_var=10,
                         n_obj=2,
                         nt=nt,
                         taut=taut)
        self.xl = -2.0 * np.ones(self.n_var)
        self.xl[0] = 0.0
        self.xu = 2.0 * np.ones(self.n_var)
        self.xu[0] = 1.0

    def get_current_t(self, t):
        self.tau = t
        t = 1 / self.nt
        t = t * floor(self.tau / self.taut)
        return t

    def _evaluate(self, X, out, *args, t=1, **kwargs):
        self.t = self.get_current_t(t)
        f1, f2 = DIMP2(X, self.t)
        out["F"] = [f1, f2]

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        pf = np.column_stack([x, 1-np.sqrt(x)])
        return pf

    def get_pf_t(self):
        return self._calc_pareto_front()


class HE2b(DynamicMOP):
    """HE2 dynamic benchmark problem
    """
    def __init__(self, nt=10, taut=200):
        super().__init__(n_var=30,
                         n_obj=2,
                         nt=nt,
                         taut=taut)

    def get_current_t(self, t):
        self.tau = t
        t = 1 / self.nt
        t = t * floor(self.tau / self.taut)
        return t

    def _evaluate(self, X, out, *args, t=1, **kwargs):
        self.t = self.get_current_t(t)
        f1, f2 = HE2(X, self.t)
        out["F"] = [f1, f2]

    def _calc_pareto_front(self, n_pareto_points=100):
        H = 0.75 * sin(0.5 * pi * self.t) + 1.25  # time may be incorrect
        x = np.linspace(0, 1, n_pareto_points)
        f2 = (1-np.power(np.sqrt(x), H))-(np.power(x, H)*np.sin(0.5*pi*x))
        pf = anp.column_stack([x, f2])
        return pf

    def get_pf_t(self):
        return self._calc_pareto_front()


class HE7b(DynamicMOP):
    """HE7 dynamic benchmark problem
    """
    def __init__(self, nt=10, taut=200):
        super().__init__(n_var=10,
                         n_obj=2,
                         nt=nt,
                         taut=taut)

        self.xl = -1 * np.ones(self.n_var)
        self.xl[0] = 0.0
        self.xu = 1 * np.ones(self.n_var)
        self.xu[0] = 1.0

    def get_current_t(self, t):
        self.tau = t
        t = 1 / self.nt
        t = t * floor(self.tau / self.taut)
        return t

    def _evaluate(self, X, out, *args, t=1, **kwargs):
        self.t = self.get_current_t(t)
        f1, f2 = HE7(X, self.t)
        out["F"] = [f1, f2]

    def _calc_pareto_front(self, n_pareto_points=100):
        H = 0.75*sin(0.5*pi*self.t)+1.25  # current time may be incorrect
        x = np.linspace(0, 1, n_pareto_points)
        f2 = (2-np.sqrt(x))*(1-np.power(x/(2-np.sqrt(x)),H))
        pf = anp.column_stack([x, f2])
        return pf

    def get_pf_t(self):
        return self._calc_pareto_front()


class HE9b(DynamicMOP):
    """HE9 dynamic benchmark problem
    """
    def __init__(self,nt =10, taut= 200):
        super().__init__(n_var=10,
                         n_obj=2,
                         nt=nt,
                         taut=taut)

    def get_current_t(self, t):
        self.tau = t
        t = 1 / self.nt
        t = t * floor(self.tau / self.taut)
        return t

    def _evaluate(self, X, out, *args, t=1, **kwargs):
        self.t = self.get_current_t(t)
        f1, f2 = HE9(X, self.t)
        out["F"] = [f1, f2]

    def _calc_pareto_front(self, n_pareto_points=100):
        H = 0.75 * sin(0.5 * pi * self.t) + 1.25 # current time may be incorrect
        x = np.linspace(0, 1, n_pareto_points)
        f2 = (2-np.sqrt(x))*(1-np.power(x/(2-np.sqrt(x)),H))
        pf = anp.column_stack([x, f2])
        return pf

    def get_pf_t(self):
        return self._calc_pareto_front()


