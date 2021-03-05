import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from utility.norm_error import *

from algorithms.fdsa import FDSA
from algorithms.spsa import SPSA
from algorithms.cs_fdsa import CsFDSA
from algorithms.cs_spsa import CsSPSA

from objectives.quad_exp import QuadExp

###
today = date.today()
p = 10
eta = np.array([1.1025, 1.6945, 1.4789, 1.9262, 0.7505,
                1.3267, 0.8428, 0.7247, 0.7693, 1.3986])

quad_exp_model = QuadExp(eta)
quad_exp_model.get_theta_star()
theta_star = quad_exp_model.theta_star
loss_star = quad_exp_model.loss_true(theta_star)

def loss_noisy(theta):
    return quad_exp_model.loss_noisy(theta)

def loss_true(theta):
    return quad_exp_model.loss_true(theta)

theta_0 = np.ones(p)
loss_0 = quad_exp_model.loss_true(theta_0)

# parameters
a = 0.02; c = 0.2; A = 100
alpha = 0.602; gamma = 0.151
iter_num = 1000; rep_num = 20

print("running FDSA")
FDSA_solver = FDSA(a=a, c=c, A=A, alpha=alpha, gamma=gamma,
                   iter_num=int(iter_num/(2*p)), rep_num=rep_num,
                   theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy)
FDSA_solver.train()
# loss_ks = np.mean(FDSA_solver.loss_ks, axis=1)
# plt.plot(loss_ks)

print("running SPSA")
SPSA_solver = SPSA(a=a, c=c, A=A, alpha=alpha, gamma=gamma,
                   iter_num=int(iter_num/2), rep_num=rep_num,
                   theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy)
SPSA_solver.train()
# loss_ks = np.mean(SPSA_solver.loss_ks, axis=1)
# plt.plot(loss_ks)

print("running CsFDSA")
CsFDSA_solver = CsFDSA(a=a, c=c, A=A, alpha=alpha, gamma=gamma,
                       iter_num=int(iter_num/p), rep_num=rep_num,
                       theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy)
CsFDSA_solver.train()
# loss_ks = np.mean(CsFDSA_solver.loss_ks, axis=1)
# plt.plot(loss_ks)

print("running CsSPSA")
CsSPSA_solver = CsSPSA(a=a, c=c, A=A, alpha=alpha, gamma=gamma,
                       iter_num=iter_num, rep_num=rep_num,
                       theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy)
CsSPSA_solver.train()
# loss_ks = np.mean(CsSPSA_solver.loss_k_all, axis=1)
# plt.plot(loss_ks)

### plot ###
# FDSA
FDSA_loss_error = get_norm_loss_error(FDSA_solver.loss_ks, loss_0, loss_star, 2*p)
FDSA_theta_error = get_norm_theta_error(FDSA_solver.theta_ks, theta_0, theta_star, 2*p)
# SPSA
SPSA_loss_error = get_norm_loss_error(SPSA_solver.loss_ks, loss_0, loss_star, 2)
SPSA_theta_error = get_norm_theta_error(SPSA_solver.theta_ks, theta_0, theta_star, 2)
# CsFDSA
CsFDSA_loss_error = get_norm_loss_error(CsFDSA_solver.loss_ks, loss_0, loss_star, p)
CsFDSA_theta_error = get_norm_theta_error(CsFDSA_solver.theta_ks, theta_0, theta_star, p)
# CsSPSA
CsSPSA_loss_error = get_norm_loss_error(CsSPSA_solver.loss_ks, loss_0, loss_star, 1)
CsSPSA_theta_error = get_norm_theta_error(CsSPSA_solver.theta_ks, theta_0, theta_star, 1)

# plot loss error
plt.figure(); plt.grid()
plt.plot(FDSA_loss_error, 'k:')
plt.plot(CsFDSA_loss_error, 'k-.')
plt.plot(SPSA_loss_error, 'k--', dashes=(5,5))
plt.plot(CsSPSA_loss_error, 'k-')

plt.xlim(xmin=0, xmax=iter_num)
plt.yscale("log")
plt.legend(["FDSA", "CS-FDSA", "SPSA", "CS-SPSA"], loc="upper right")
plt.xlabel("Number of Function Measurements")
plt.ylabel("Normalized Errors in Loss Function")
plt.savefig('figures/quad-exp-loss-p-' + str(p) + '-' + str(today) + '.pdf')

# plot theta error
plt.figure(); plt.grid()
plt.plot(FDSA_theta_error, 'k:')
plt.plot(CsFDSA_theta_error, 'k-.')
plt.plot(SPSA_theta_error, 'k--', dashes=(5,5))
plt.plot(CsSPSA_theta_error, 'k-')

plt.xlim(xmin=0, xmax=iter_num)
plt.yscale("log")
plt.legend(["FDSA", "CS-FDSA", "SPSA", "CS-SPSA"], loc="upper right")
plt.xlabel("Number of Function Measurements")
plt.ylabel(r"Normalized Error in Estimation of $\mathbf{\theta}$")
plt.savefig('figures/quad-exp-theta-p-' + str(p) + '-' + str(today) + '.pdf')

print("finish plotting!")