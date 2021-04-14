from datetime import date

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from objectives.skewed_quartic import SkewedQuartic

from algos.spsa import SPSA
from algos.mspsa import MSPSA

from utils.norm_error import *

### init obj ###
p = 100
loss_obj = SkewedQuartic(p)
theta_star = loss_obj.theta_star
loss_star = loss_obj.loss_truth(theta_star)

### init algo ###
theta_0 = np.ones(p) * 5
loss_0 = loss_obj.loss_truth(theta_0)
meas_num = 200
rep_num = 1

### SPSA ###
print('Running SPSA')
solver_SPSA = SPSA(
    a=0.1, c=0.5, A=500, alpha=0.7, gamma=0.167,
    iter_num=int(meas_num/2), rep_num=rep_num,
    theta_0=theta_0, loss_obj=loss_obj)

# solver_SPSA.train()
# SPSA_loss_error = get_norm_loss_error(solver_SPSA.loss_ks, loss_0, loss_star, multi=2)
# SPSA_theta_error = get_norm_theta_error(solver_SPSA.theta_ks, theta_0, theta_star, multi=2)
# fig = plt.figure()
# plt.plot(SPSA_loss_error)
# plt.show()
# plt.close()

### MSPSA ###
print('Running MSPSA')
d = p // 2
solver_MSPSA = MSPSA(
    d=d, a=0.1, c=0.5, A=500, alpha=0.7, gamma=0.167,
    iter_num=int(meas_num/2), rep_num=rep_num,
    theta_0=theta_0, loss_obj=loss_obj)
solver_MSPSA.train()
MSPSA_loss_error = get_norm_loss_error(solver_MSPSA.loss_ks, loss_0, loss_star, multi=2)
MSPSA_theta_error = get_norm_theta_error(solver_MSPSA.theta_ks, theta_0, theta_star, multi=2)
fig = plt.figure()
plt.plot(MSPSA_loss_error)
plt.show()
plt.close()


raise

today = date.today()



### Random Search ###
print('Running Random Search')
RS_solver = RandomSearch(sigma=0.1,
                         iter_num=meas_num, rep_num=rep_num,
                         d=d, theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy,
                         record_theta_flag=True, record_loss_flag=True)
RS_solver.train()
RS_loss_error = get_norm_loss_error(RS_solver.loss_ks, loss_0, loss_star, multi=1)
RS_theta_error = get_norm_theta_error(RS_solver.theta_ks, theta_0, theta_star, multi=1)
# plt.plot(RS_loss_error)

### Stochastic Ruler ###
print('Running Stochastic Ruler')
M_multiplier = 0.5
SR_solver = StochasticRuler(M_multi=0.5, meas_num=meas_num, rep_num=rep_num,
                            d=d, theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy,
                            record_theta_flag=True, record_loss_flag=True)
SR_solver.train()
SR_M_ks = SR_solver.M_ks
SR_loss_error = get_norm_loss_error(SR_solver.loss_ks, loss_0, loss_star, multi=SR_M_ks)[:meas_num+1]
SR_theta_error = get_norm_theta_error(SR_solver.theta_ks, theta_0, theta_star, multi=SR_M_ks)[:meas_num+1]
plt.plot(SR_loss_error)

### Plot ###
matplotlib.rcParams.update({"font.size": 12})
linewidth = 2

# plot theta
plot_theta = plt.figure()
plt.grid()
plt.title(r'Normalized Mean-Squared Error for $\hat{\mathbf{\theta}}_k$')
plt.xlabel("Number of Loss Function Measurements")
plt.ylabel("Normalized Mean-Squared Error")
plt.ylim(0, 1)

plt.plot(RS_theta_error**2, linewidth=linewidth, linestyle=":", color="black")
plt.plot(SR_theta_error**2, linewidth=linewidth, linestyle="-", color="black")
plt.plot(MSPSA_theta_error**2, linewidth=linewidth, linestyle="--", color="black")

plt.legend(["Local Random Search", "Stochastic Ruler", "MSPSA"])
plt.close()
plot_theta.savefig("figures/skewed-quartic-theta-error-" + str(today) + ".pdf", bbox_inches='tight')

# plot loss
plot_loss = plt.figure()
plt.grid()
plt.title("Normalized Error for Loss")
plt.xlabel("Number of Loss Function Measurements")
plt.ylabel("Normalized Error")
plt.ylim(0, 1)

plt.plot(RS_loss_error, linewidth=linewidth, linestyle=":", color="black")
plt.plot(SR_loss_error, linewidth=linewidth, linestyle="-", color="black")
plt.plot(MSPSA_loss_error, linewidth=linewidth, linestyle="--", color="black")

plt.legend(["Local Random Search", "Stochastic Ruler", "MSPSA"])
plt.close()
plot_loss.savefig("figures/skewed-quartic-loss-error-" + str(today) + ".pdf", bbox_inches='tight')