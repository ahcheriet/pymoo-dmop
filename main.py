from dynamic_mop import *
from dynamic_algorithms import DNSGA2_a
import autograd.numpy as anp
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt
from pymoo.factory import get_performance_indicator

# nt: severity of change
# taut: frequency of change
# tauT: maximum number of generation
# tau : current generation
# examples of nt and taut values
nt_ = 20
taut_ = 300
tauT = 600


algorithm = DNSGA2_a(pop_size=200)
problem = DMOP2(nt=nt_, taut=taut_)

POF = []
PF = []
change_time = []


# This call back is to store the result(pf) in each time the problem changes
def callback(ex_algorithm):
    if ex_algorithm.has_changed:
        POF.append(ex_algorithm.problem.get_pf_t())  # Pareto Optimal front at time t
        change_time.append(ex_algorithm.problem.tau)
        print("Changed")


algorithm.callback = callback


res = minimize(problem,
               algorithm,
               ("n_gen", tauT),
               save_history=True,
               verbose=False,
               seed=anp.random.randint(1,1000))

n_gen = []    # corresponding number of function evaluations\
F = []          # the objective space values in each generation
cv = []         # constraint violation in each generation


for algorithm in res.history:
    n_gen.append(algorithm.n_gen)
    opt = algorithm.opt
    cv.append(opt.get("CV").min())
    feasible = anp.where(opt.get("feasible"))[0]
    _F = opt.get("F")[feasible]
    F.append(_F)

igd = []
# this is to calculate the IGD every time the problem changes.

for idx, pf in enumerate(POF):
    normalize = False
    metric = get_performance_indicator("igd", pf)
    igd = igd + [metric.do(f) for f in F[idx*taut_:(idx*taut_+taut_)]]
print(anp.mean(igd))
plt.plot(n_gen, igd, '-o', markersize=4, linewidth=2, color="green")
plt.yscale("log")          # enable log scale if desired
plt.title("Convergence")
plt.xlabel("Iteration")
plt.ylabel("IGD")
plt.show()

# just plotting the POFs
plot = Scatter()

for pf in POF:
    plot.add(pf, plot_type="line", color="black", linewidth=2)

for idx in change_time:
    print(idx,len(F))
    pf = F[idx-1]
    plot.add(pf, color='red')

# example
# for pf in PF:
#     plot.add(pf, plot_type="points", color='red')

plot.show()

