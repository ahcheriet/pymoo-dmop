import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.population import Population
from pymoo.docs import parse_doc_string
from pymoo.util.misc import has_feasible
from copulas.multivariate import VineCopula
from pymoo.factory import get_reference_directions
from scipy.special import kl_div
from scipy.spatial import cKDTree as KDTree

ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

# create the algorithm object

class EMT_CEDA(NSGA2):

    def __init__(self,
                 k=5,
                 dynamic=True,
                 **kwargs):
        """

        Parameters
        ----------
        """

        super().__init__( ref_dirs=ref_dirs,**kwargs)

        self.dynamic = dynamic

    def _infill(self):
        pop, len_pop, len_off = self.pop, self.pop_size, self.n_offsprings
        xl, xu = self.problem.bounds()
        X = pop.get("X")
        #X = self.opt.get("X")
        data = pd.DataFrame(X)
#        center = VineCopula('center')
        regular = VineCopula('regular')
#        direct = VineCopula('direct')

#        center.fit(data)
        regular.fit(data, truncated=5)
#        direct.fit(data)

#        center_samples = center.sample(1000)
        Sample = regular.sample(100)
#        direct_samples = direct.sample(1000)
        Xp= Sample.to_numpy()

        # create the population to proceed further
        off = Population.new(X=Xp)

        return off

    def _advance(self, infills=None, **kwargs):

        if self.dynamic:
            I = np.random.permutation(len(self.pop))[:10]
            pop_copy = Population.new(X=self.pop[I].get("X"))
            self.evaluator.eval(self.problem, pop_copy, t=self.n_gen, skip_already_evaluated=self.dynamic)

            # detect change
            delta = np.abs(pop_copy.get("F") - self.pop[I].get("F")).mean()
            if delta != 0:
                self.problem.has_changed = True
            else:
                self.problem.has_changed = False

        if infills is not None:
            self.pop = Population.merge(self.pop, infills)
            self.evaluator.eval(self.problem, self.pop, t=self.n_gen, skip_already_evaluated=self.dynamic)

        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(self.problem, self.pop, n_survive=self.pop_size, algorithm=self)


class RM_MEDA(NSGA2):

    def __init__(self,
                 k=5,
                 dynamic=True,
                 **kwargs):
        """
        Regularity Model-Based Multiobjective Estimation of Distribution Algorithm (RM-MEDA)

        Parameters
        ----------
        n_offsprings : int
            The number of individuals created in each iteration.
        pop_size : int
            The number of individuals which are surviving from the offspring population (non-elitist)
        k: int
            Parameter of the rm_meda algorithm number of cluster
        """

        super().__init__(**kwargs)

        self._K = k
        self.dynamic = dynamic
        self.Models = []
        self.probabilities = []
        self.tracks = []

    def _infill(self):
        pop, len_pop, len_off = self.pop, self.pop_size, self.n_offsprings
        xl, xu = self.problem.bounds()
        X = pop.get("X")
        # Modeling
        Model, probability = LocalPCA(X, self.problem.n_obj, self._K)
        # Sampling
        Xp = RMMEDA_operator(X, len(X), self._K, self.problem.n_obj, Model, probability, xl, xu)

        # create the population to proceed further
        off = Population.new(X=Xp)

        return off

    def _advance(self, infills=None, **kwargs):

        if self.dynamic:
            I = np.random.permutation(len(self.pop))[:10]
            pop_copy = Population.new(X=self.pop[I].get("X"))
            self.evaluator.eval(self.problem, pop_copy, t=self.n_gen, skip_already_evaluated=not self.dynamic)

            # detect change
            delta = np.abs(pop_copy.get("F") - self.pop[I].get("F")).mean()
            if self.dynamic:
                if delta != 0:
                    self.problem.has_changed = True
                else:
                    self.problem.has_changed = False

        if infills is not None:
            self.pop = Population.merge(self.pop, infills)
            # if self.problem.has_changed:
            #     self.pop = Reinitialize Randomly
            self.evaluator.eval(self.problem, self.pop, t=self.n_gen, skip_already_evaluated=not self.dynamic)

        # execute the survival to find the fittest solutions, selecting P_{t+1} using NDS
        self.pop = self.survival.do(self.problem, self.pop, n_survive=self.pop_size, algorithm=self)


    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]



class ETM_RM_MEDA(NSGA2):

    def __init__(self,
                 samples_t=20,
                 nbr_previous=4,
                 k=5,
                 dynamic=True,
                 **kwargs):
        """
        Regularity Model-Based Multiobjective Estimation of Distribution Algorithm (RM-MEDA)

        Parameters
        ----------
        n_offsprings : int
            The number of individuals created in each iteration.
        pop_size : int
            The number of individuals which are surviving from the offspring population (non-elitist)
        k: int
            Parameter of the rm_meda algorithm number of cluster
        """

        super().__init__(**kwargs)

        self._K = k
        self.dynamic = dynamic
        self.Models = []
        self.probabilities = []
        self.tracks = []
        self.samples_t = samples_t
        self.nbr_previous = nbr_previous

    def get_best_model(self, models, probability, Y):
        kl_divs = []
        XX = []
        track = self.tracks
        xl, xu = self.problem.bounds()
        Xp = RMMEDA_operator(Y, len(Y), self._K, self.problem.n_obj, models[0], probability, xl, xu)
        if len(self.tracks)>self.nbr_previous:
            Slice = -1*self.nbr_previous
            track = self.tracks[Slice:]
        for i in self.tracks:
            X_tmp = RMMEDA_operator(Y, self.samples_t, self._K, self.problem.n_obj, self.Models[i-1], self.probabilities[i-1], xl, xu)
            XX.append(X_tmp)
            kl_val = kl_div(X_tmp, Y)
            ccc = np.nan_to_num(kl_val, posinf=10, neginf=-10)
            val = sum(np.average(ccc, axis=0))

            kl_divs.append(val)
        if len(kl_divs)<1:
            return Xp
        else:
            index_max = np.argmax(kl_divs)
            index_min = np.argmin(kl_divs)
        return np.concatenate([XX[index_max],XX[index_min]])

    def _infill(self):
        pop, len_pop, len_off = self.pop, self.pop_size, self.n_offsprings
        xl, xu = self.problem.bounds()
        X = pop.get("X")
        # Modeling
        Model, probability = LocalPCA(X, self.problem.n_obj, self._K)
        self.Models.append(Model)
        self.probabilities.append(probability)
        # Sampling
        Xp = RMMEDA_operator(X, len(X), self._K, self.problem.n_obj, Model, probability, xl, xu)

        # create the population to proceed further
        off = Population.new(X=Xp)
        track = self.tracks
        if len(self.tracks)>self.nbr_previous:
            Slice = -1*self.nbr_previous
            track = self.tracks[Slice:]
        for i in track:
            X_tmp = RMMEDA_operator(X, self.samples_t, self._K, self.problem.n_obj, self.Models[i-1], self.probabilities[i-1], xl, xu)
            off_tmp = Population.new(X=X_tmp)
            off = Population.merge(off_tmp, off)
        off = Population.new(X=self.get_best_model([Model], probability, X))
        return off

    def _advance(self, infills=None, **kwargs):

        if self.dynamic:
            I = np.random.permutation(len(self.pop))[:10]
            pop_copy = Population.new(X=self.pop[I].get("X"))
            self.evaluator.eval(self.problem, pop_copy, t=self.n_gen, skip_already_evaluated=not self.dynamic)

            # detect change
            delta = np.abs(pop_copy.get("F") - self.pop[I].get("F")).mean()
            if self.dynamic:
                if delta != 0:
                    self.problem.has_changed = True
                    self.tracks.append(self.problem.tau)
                else:
                    self.problem.has_changed = False

        if infills is not None:
            self.pop = Population.merge(self.pop, infills)
            self.evaluator.eval(self.problem, self.pop, t=self.n_gen, skip_already_evaluated=not self.dynamic)

        # execute the survival to find the fittest solutions, selecting P_{t+1} using NDS
        self.pop = self.survival.do(self.problem, self.pop, n_survive=self.pop_size, algorithm=self)


    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]



def RMMEDA_operator(PopDec, n_ind, K, M, Model, probability, XLow, XUpp):
    N, D = PopDec.shape
    ## Modeling
    ## Reproduction
#    N = n_ind
    OffspringDec = np.zeros((N, D))
    # Generate new trial solutions one by one
    for i in np.arange(n_ind):
        # Select one cluster by Roulette-wheel selection
        k = (np.where(np.random.rand() <= probability))[0][0]
        # Generate one offspring
        if not len(Model[k]['eVector']) == 0:
            lower = Model[k]['a'] - 0.25 * (Model[k]['b'] - Model[k]['a'])
            upper = Model[k]['b'] + 0.25 * (Model[k]['b'] - Model[k]['a'])
            trial = np.random.uniform(0, 1) * (upper - lower) + lower  # ,(1,M-1)
            sigma = np.sum(np.abs(Model[k]['eValue'][M - 1:D])) / (D - M + 1)
            OffspringDec[i, :] = Model[k]['mean'] + trial * Model[k]['eVector'][:,
                                                            :M - 1].conj().transpose() + np.random.randn(D) * np.sqrt(
                sigma)
        else:
            OffspringDec[i, :] = Model[k]['mean'] + np.random.randn(D)
        NN, D = OffspringDec.shape
        low = np.tile(XLow, (NN, 1))
        upp = np.tile(XUpp, (NN, 1))
        lbnd = OffspringDec <= low
        ubnd = OffspringDec >= upp
        OffspringDec[lbnd] = 0.5 * (PopDec[lbnd] + low[lbnd])
        OffspringDec[ubnd] = 0.5 * (PopDec[ubnd] + upp[ubnd])

    return OffspringDec


def LocalPCA(PopDec, M, K, max_iter=50):
    N, D = np.shape(PopDec)  # Dimensions
    Model = [{'mean': PopDec[k],  # The mean of the model
              'PI': np.eye(D),  # The matrix PI
              'eVector': [],  # The eigenvectors
              'eValue': [],  # The eigenvalues
              'a': [],  # The lower bound of the projections
              'b': []} for k in range(K)]  # The upper bound of the projections

    # Modeling
    for iteration in range(1, max_iter):
        # Calculate the distance between each solution and its projection in
        # affine principal subspace of each cluster
        distance = np.zeros((N, K))  # matrix of zeros N*K
        for k in range(K):
            distance[:, k] = np.sum((PopDec - np.tile(Model[k]['mean'], (N, 1))).dot(Model[k]['PI']) * (
                        PopDec - np.tile(Model[k]['mean'], (N, 1))), 1)
        # Partition
        partition = np.argmin(distance, 1)  # get the index of mins
        # Update the model of each cluster
        updated = np.zeros(K, dtype=bool)  # array of k false
        for k in range(K):
            oldMean = Model[k]['mean']
            current = partition == k
            if sum(current) < 2:
                if not any(current):
                    current = [np.random.randint(N)]
                Model[k]['mean'] = PopDec[current, :]
                Model[k]['PI'] = np.eye(D)
                Model[k]['eVector'] = []
                Model[k]['eValue'] = []
            else:
                Model[k]['mean'] = np.mean(PopDec[current, :], 0)
                cc = np.cov((PopDec[current, :] - np.tile(Model[k]['mean'], (np.sum(current), 1))).T)
                eValue, eVector = np.linalg.eig(cc)
                rank = np.argsort(-(eValue), axis=0)
                eValue = -np.sort(-(eValue), axis=0)
                Model[k]['eValue'] = np.real(eValue).copy()
                Model[k]['eVector'] = np.real(eVector[:, rank]).copy()
                Model[k]['PI'] = Model[k]['eVector'][:, (M - 1):].dot(
                    Model[k]['eVector'][:, (M - 1):].conj().transpose())

            updated[k] = not any(current) or np.sqrt(np.sum((oldMean - Model[k]['mean']) ** 2)) > 1e-5

        # Break if no change is made
        if not any(updated):
            break

    # Calculate the smallest hyper-rectangle of each model
    for k in range(K):
        if len(Model[k]['eVector']) != 0:
            hyperRectangle = (PopDec[partition == k, :] - np.tile(Model[k]['mean'], (sum(partition == k), 1))).dot(
                Model[k]['eVector'][:, 0:M - 1])
            Model[k]['a'] = np.min(hyperRectangle)  # this should by tested
            Model[k]['b'] = np.max(hyperRectangle)  # this should by tested
        else:
            Model[k]['a'] = np.zeros((1, M - 1))
            Model[k]['b'] = np.zeros((1, M - 1))

    # Calculate the probability of each cluster for reproduction
    # Calculate the volume of each cluster
    volume = np.array([Model[k]['b'] for k in range(K)]) - np.array(
        [Model[k]['a'] for k in range(K)])  # this should be tested
    #    volume = prod(cat(1,Model.b)-cat(1,Model.a),2)
    # Calculate the cumulative probability of each cluster
    probability = np.cumsum(volume / np.sum(volume))

    return Model, probability

# parse_doc_string(rm_meda.__init__)
# https://mail.python.org/pipermail/scipy-user/2011-May/029521.html

def KLdivergence(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate samples.

  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.

  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).

  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
  """

  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n,d = x.shape
  m,dy = y.shape

  assert(d == dy)


  # Build a KD tree representation of the samples and find the nearest neighbour
  # of each point in x.
  xtree = KDTree(x)
  ytree = KDTree(y)

  # Get the first two nearest neighbours for x, since the closest one is the
  # sample itself.
  r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
  s = ytree.query(x, k=1, eps=.01, p=2)[0]

  # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
  # on the first term of the right hand side.
  return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))