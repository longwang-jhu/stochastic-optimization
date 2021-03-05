import numpy as np
import matplotlib.pyplot as plt

import loss.quadratic_exponential

from algorithm.fdsa import FDSA
from algorithm.spsa import SPSA
from algorithm.cs_fdsa import CsFDSA
from algorithm.cs_spsa import CsSPSA

p = 10
eta = np.array([1.1025, 1.6945, 1.4789, 1.9262, 0.7505,
                1.3267, 0.8428, 0.7247, 0.7693, 1.3986])
theta_star = np.array([0.286, 0.229, 0.248, 0.211, 0.325,
                        0.263, 0.315, 0.327, 0.323, 0.256])
loss_star = loss.quadratic_exponential.loss_true(eta, theta_star)

# p = 100
# np.random.seed(100)
# eta = np.random.uniform(low=0.5, high=2.0, size=p)
# import scipy.optimize
# def grad(theta):
#     return loss.quadratic_exponential.gradient_true(eta, theta)
# theta_star = scipy.optimize.broyden1(grad, np.ones(p), f_tol=1e-14)
# loss_star = loss.quadratic_exponential.loss_true(eta, theta_star)

def loss_noisy(theta):
    X = np.random.exponential(1 / eta)
    return loss.quadratic_exponential.loss_noisy(X, theta)

def loss_true(theta):
    return loss.quadratic_exponential.loss_true(eta, theta)

theta_0 = np.ones(p)
loss_0 = loss.quadratic_exponential.loss_true(eta, theta_0)

# parameters
a = 0.02; c = 0.2; A = 100
alpha = 0.602; gamma = 0.151
iter_num = 50000; rep_num = 20

print("running FDSA")
FDSA_solver = FDSA(a=a, c=c, A=A, alpha=alpha, gamma=gamma,
                   iter_num=int(iter_num/(2*p)), rep_num=rep_num,
                   theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy)
FDSA_solver.train()
# loss_ks = np.mean(FDSA_solver.loss_k_all, axis=1)
# plt.plot(loss_ks)

print("running SPSA")
SPSA_solver = SPSA(a=a, c=c, A=A, alpha=alpha, gamma=gamma,
                   iter_num=int(iter_num/2), rep_num=rep_num,
                   theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy)
SPSA_solver.train()
# loss_ks = np.mean(SPSA_solver.loss_k_all, axis=1)
# plt.plot(loss_ks)

print("running CS-FDSA")
CS_FDSA_solver = CsFDSA(a=a, c=c, A=A, alpha=alpha, gamma=gamma,
                   iter_num=int(iter_num/p), rep_num=rep_num,
                   theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy)
CS_FDSA_solver.train()
# loss_ks = np.mean(CS_FDSA_solver.loss_k_all, axis=1)
# plt.plot(loss_ks)

print("running CS-SPSA")
CS_SPSA_solver = CsSPSA(a=a, c=c, A=A, alpha=alpha, gamma=gamma,
                   iter_num=iter_num, rep_num=rep_num,
                   theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy)
CS_SPSA_solver.train()
# loss_ks = np.mean(CS_SPSA_solver.loss_k_all, axis=1)
# plt.plot(loss_ks)


def get_normalize_loss_error(loss_all, loss_0, loss_star, multiplier):
    loss_ks = np.mean(loss_all, axis=1) # average over replicates
    loss_ks_error = (loss_ks-loss_star) / (loss_0-loss_star)
    return np.repeat(loss_ks_error, multiplier)

def get_normalize_theta_error(theta_all, theta_0, theta_star, multiplier):
    theta_ks_error = np.linalg.norm(theta_all - theta_star[:,None,None], axis=0)
    theta_ks_error = np.mean(theta_ks_error, axis=1) / np.linalg.norm(theta_0 - theta_star)
    return np.repeat(theta_ks_error, multiplier)

### plot ###
# FDSA
FDSA_loss_ks_error = get_normalize_loss_error(FDSA_solver.loss_k_all, loss_0, loss_star, 2*p)
FDSA_theta_ks_error = get_normalize_theta_error(FDSA_solver.theta_k_all, theta_0, theta_star, 2*p)

# SPSA
SPSA_loss_ks_error = get_normalize_loss_error(SPSA_solver.loss_k_all, loss_0, loss_star, 2)
SPSA_theta_ks_error = get_normalize_theta_error(SPSA_solver.theta_k_all, theta_0, theta_star, 2)

# CS_FDSA
CS_FDSA_loss_ks_error = get_normalize_loss_error(CS_FDSA_solver.loss_k_all, loss_0, loss_star, p)
CS_FDSA_theta_ks_error = get_normalize_theta_error(CS_FDSA_solver.theta_k_all, theta_0, theta_star, p)

# CS_SPSA
CS_SPSA_loss_ks_error = get_normalize_loss_error(CS_SPSA_solver.loss_k_all, loss_0, loss_star, 1)
CS_SPSA_theta_ks_error = get_normalize_theta_error(CS_SPSA_solver.theta_k_all, theta_0, theta_star, 1)

plt.figure()
plt.grid()

plt.plot(np.concatenate(([1], FDSA_loss_ks_error)), 'k:')
plt.plot(np.concatenate(([1], CS_FDSA_loss_ks_error)), 'k-.')
plt.plot(np.concatenate(([1], SPSA_loss_ks_error)), 'k--', dashes=(5,5))
plt.plot(np.concatenate(([1], CS_SPSA_loss_ks_error)), 'k-')

plt.xlim(xmin=0, xmax=iter_num)
plt.yscale("log")
plt.legend(["FDSA", "CS-FDSA", "SPSA", "CS-SPSA"], loc="upper right")
plt.xlabel("Number of Function Measurements")
plt.ylabel("Normalized Errors in Loss Function")
plt.savefig('quadatic_exponential_p_10_loss_2020_09_15.pdf')

plt.figure()
plt.grid()
plt.plot(np.concatenate(([1], FDSA_theta_ks_error)), 'k:')
plt.plot(np.concatenate(([1], CS_FDSA_theta_ks_error)), 'k-.')
plt.plot(np.concatenate(([1], SPSA_theta_ks_error)), 'k--', dashes=(5,5))
plt.plot(np.concatenate(([1], CS_SPSA_theta_ks_error)), 'k-')

plt.xlim(xmin=0, xmax=iter_num)
plt.yscale("log")
plt.legend(["FDSA", "CS-FDSA", "SPSA", "CS-SPSA"], loc="upper right")
plt.xlabel("Number of Function Measurements")
plt.ylabel(r"Normalized Error in Estimation of $\mathbf{\theta}$")
plt.savefig('quadatic_exponential_p_10_theta_2020_09_15.pdf')

print("finish plotting")