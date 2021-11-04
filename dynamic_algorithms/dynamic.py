import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from pymoo.docs import parse_doc_string
from pymoo.core.survival import Survival
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.dominator import Dominator
from pymoo.util.misc import find_duplicates, has_feasible
from pymoo.util.termination.default import MultiObjectiveDefaultTermination

# TODO implement an abstract class for the dynamic algorithm


class DynamicAlgorithm(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
           #      selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SimulatedBinaryCrossover(eta=15, prob=0.9),
                 mutation=PolynomialMutation(prob=None, eta=20),
        #         survival=RankAndCrowdingSurvival(),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

        Parameters
        ----------
        pop_size : {pop_size}
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
#                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
 #                        survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         display=display,
                         advance_after_initial_infill=True,
                         **kwargs)
        self.default_termination = MultiObjectiveDefaultTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'

    def detect_change(self, *args, **kwargs):
        # an example of detecting a change in the problem, we can implement anther methods.
        I = np.random.permutation(len(self.pop))[:10]
        pop_copy = Population.new(X=self.pop[I].get("X"))
        self.evaluator.eval(self.problem, pop_copy, t=self.n_gen, skip_already_evaluated=False)
        # detect change
        delta = np.abs(pop_copy.get("F") - self.pop[I].get("F")).mean()
        return delta != 0

    def _infill(self):
        # the infill is not implemented
        pass

    def _advance(self, infills=None, **kwargs):
        # here we implement the advance method
        if self.detect_change():
            self.problem.has_changed = True
            self.react_to_change()
        else:
            self.problem.has_changed = True
        pass

    def react_to_change(self, *args, **kwargs):
        pass

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]


