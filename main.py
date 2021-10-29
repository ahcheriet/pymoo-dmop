import numpy as np

from dynamic_mop import *
from dynamic_algorithms import DNSGA2_a, calculate_MIGD, RM_MEDA
import autograd.numpy as anp
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator


# nt: severity of change
# taut: frequency of change
# tauT: maximum number of generation
# tau : current generation
# examples of nt and taut values
nt_ = 5
taut_ = 10
tauT = 200

evaluator = Evaluator(skip_already_evaluated=True)
evaluator1 = Evaluator(skip_already_evaluated=True)

algorithm = DNSGA2_a(pop_size=200)
algorithm2 = RM_MEDA(pop_size=200, evaluator = evaluator1)
algorithm3 = NSGA2(pop_size=200, evaluator = evaluator)


problem = DMOP2(nt=nt_, taut=taut_)


res = minimize(problem,
               algorithm,
               ("n_gen", tauT),
               verbose=False,
               callback=calculate_MIGD(),
               seed=2)
res2 = minimize(problem,
               algorithm2,
               ("n_gen", tauT),
               verbose=False,
               callback=calculate_MIGD(),
               seed=2)
res3 = minimize(problem,
               algorithm3,
               ("n_gen", tauT),
               verbose=False,
               callback=calculate_MIGD(),
               seed=2)


print(res.algorithm.callback.data["MIGD"])
print(res2.algorithm.callback.data["MIGD"])
print(res3.algorithm.callback.data["MIGD"])

plt.plot(np.arange(tauT), res.algorithm.callback.data["igd"], '-o', markersize=4, linewidth=2, color="green")
plt.plot(np.arange(tauT), res2.algorithm.callback.data["igd"], '-o', markersize=4, linewidth=2, color="red")
plt.plot(np.arange(tauT), res3.algorithm.callback.data["igd"], '-o', markersize=4, linewidth=2, color="blue")
plt.yscale("log")
plt.title("Convergence")
plt.xlabel("Iteration")
plt.ylabel("IGD")
plt.show()

# just plotting the POFs
plot = Scatter()

for pf in res.algorithm.callback.data["POF"]:
    plot.add(pf, plot_type="line", color="black", linewidth=2)


# for pf in res.algorithm.callback.data["PF"]:
#     plot.add(pf, color='red')


plot.show()

