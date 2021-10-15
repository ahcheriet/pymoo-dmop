import numpy as np

from dynamic_mop import *
from dynamic_algorithms import DNSGA2_a, calculate_MIGD
import autograd.numpy as anp
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2

# nt: severity of change
# taut: frequency of change
# tauT: maximum number of generation
# tau : current generation
# examples of nt and taut values
nt_ = 20
taut_ = 100
tauT = 200


algorithm = DNSGA2_a(pop_size=200)
#algorithm = NSGA2(pop_size=200)
problem = DMOP2(nt=nt_, taut=taut_)


res = minimize(problem,
               algorithm,
               ("n_gen", tauT),
               verbose=False,
               callback=calculate_MIGD(),
               seed=np.random.randint(1,1000))


print(res.algorithm.callback.data["MIGD"])
plt.plot(np.arange(tauT), res.algorithm.callback.data["igd"], '-o', markersize=4, linewidth=2, color="green")
plt.yscale("log")
plt.title("Convergence")
plt.xlabel("Iteration")
plt.ylabel("IGD")
plt.show()

# just plotting the POFs
plot = Scatter()

for pf in res.algorithm.callback.data["POF"]:
    plot.add(pf, plot_type="line", color="black", linewidth=2)


for pf in res.algorithm.callback.data["PF"]:
    plot.add(pf, plot_type="line", color='red')


plot.show()

