from dynamic_mop import *
from time import time
from dynamic_algorithms import DNSGA2_a, calculate_MIGD, RM_MEDA, ETM_RM_MEDA
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator
from dynamic_mop.cec2018 import *
from dynamic_mop.dmop import DMOP2
from pymoo.factory import get_problem
from pymoo.core.callback import Callback

# nt: severity of change
# taut: frequency of change
# tauT: maximum number of generation
# tau : current generation
# examples of nt and taut values
print("________Start__________")
s = time()
taut_ = 5
nt_ = 10
tauT = 100
Seed = np.random.randint(1,100)
#tauT = 3*nt_
#algorithm = EMT_CEDA(pop_size=100,dynamic=False)

algorithm = RM_MEDA(pop_size=100, dynamic=True) # green
algorithm2 = ETM_RM_MEDA(pop_size=100, samples_t=10, nbr_previous= 2, dynamic=True) # red
algorithm3 = ETM_RM_MEDA(pop_size=100, samples_t=50, nbr_previous= 20, dynamic=True) # blue
algorithm4 = DNSGA2_a(pop_size=100) # black


#problem = CEC2018(problemID='DF1', nt=nt_, taut=taut_)
#problem = get_problem("zdt1")

problem = CEC2018(problemID='DF1', nt=nt_, taut=taut_)
calculate=calculate_MIGD()
res = minimize(problem,
               algorithm,
               ("n_gen", tauT),
               verbose=True,
               callback=calculate,
               seed=Seed)
res2 = minimize(problem,
               algorithm2,
               ("n_gen", tauT),
               verbose=True,
               callback=calculate_MIGD(),
               seed=Seed)
res3 = minimize(problem,
               algorithm3,
               ("n_gen", tauT),
               verbose=True,
               callback=calculate_MIGD(),
               seed=Seed)

res4 = minimize(problem,
               algorithm4,
               ("n_gen", tauT),
               verbose=True,
               callback=calculate_MIGD(),
               seed=Seed)


print(res.algorithm.callback.data["MIGD"])
print(res2.algorithm.callback.data["MIGD"])
print(res3.algorithm.callback.data["MIGD"])
print(res4.algorithm.callback.data["MIGD"])

plt.plot(np.arange(tauT), res.algorithm.callback.data["igd"], '-o', markersize=4, linewidth=2, color="green")
plt.plot(np.arange(tauT), res2.algorithm.callback.data["igd"], '-o', markersize=4, linewidth=2, color="red")
plt.plot(np.arange(tauT), res3.algorithm.callback.data["igd"], '-o', markersize=4, linewidth=2, color="blue")
plt.plot(np.arange(tauT), res4.algorithm.callback.data["igd"], '-o', markersize=4, linewidth=2, color="black")
plt.yscale("log")
plt.title("Convergence")
plt.xlabel("Iteration")
plt.ylabel("IGD")
plt.show()

# just plotting the POFs
plot = Scatter()

# for pf in res.algorithm.callback.data["POF"]:
#     plot.add(pf, plot_type="line", color="black", linewidth=2)
#

# for pf in res.algorithm.callback.data["PF"]:
#      plot.add(pf, color='red')


# plot.show()
# e = time()
# print(e-s)
