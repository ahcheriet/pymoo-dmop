from pymoo.core.population import Population
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.algorithms.moo.nsga2 import NSGA2
import numpy as np


class DNSGA2_a(NSGA2):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.sampling = FloatRandomSampling(self)
        self.sol_to_be_tested = None
        self.sol_to_be_new = None
        self.has_changed = False
    # def _infill(self):
    #
    #     if self.sol_to_be_tested:
    #         F, G = self.sol_to_be_tested.get("F", "G")
    #         M = np.column_stack(F, G)
    #
    #         _F, _G = self.sol_to_be_new.get("F", "G")
    #         _M = np.column_stack(F, G)
    #
    #     pop_by_mating = self.mating.do(self.problem, self.pop, 20)
    #
    #     if self.change_detected:
    #
    #         pop_by_random = self.sampling.do(self.problem, 10)
    #
    #     else:
    #         pass
    #
    #
    #     new_pop = Population.merge(self.pop, pop_by_mating)
    #
    #     self.survival.do()
    #
    #     self.sol_to_be_tested = self.pop[I]
    #

    def _advance(self, **kwargs):
        off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)

        I = np.random.permutation(len(self.pop))[:10]

        pop_copy = Population.new(X=self.pop[I].get("X"))
        self.evaluator.eval(self.problem, pop_copy, t=self.n_gen, skip_already_evaluated=False)

        # detect change
        delta = np.abs(pop_copy.get("F") - self.pop[I].get("F")).mean()

        if delta != 0:
            self.problem.has_changed = True
            self.has_changed = True
            pop_by_mating = self.mating.do(self.problem, self.pop,  int(self.n_offsprings * 0.8),  algorithm=self)
            pop_by_random = self.sampling.do(problem=self.problem, n_samples=int(self.n_offsprings * 0.2))
            pop = Population.merge(pop_by_random, pop_by_mating)
            self.evaluator.eval(self.problem, pop, t=self.n_gen, skip_already_evaluated=False)
        else:
            self.has_changed = False
            self.problem.has_changed = False
            pass
        # merge the offsprings with the current population
        if off is not None:
            self.pop = Population.merge(self.pop, off)
            self.evaluator.eval(self.problem, self.pop, t=self.n_gen, skip_already_evaluated=False)

        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(self.problem, self.pop, n_survive=self.pop_size, algorithm=self)