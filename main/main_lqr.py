#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 16:41:02 2020

@author: Long Wang
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import date
today = date.today()

# import algorithms
from algorithms.spsa import SPSA
from algorithms.two_spsa import TwoSPSA
from algorithms.cs_spsa import CsSPSA

# import objective
from objectives.lqr import LQR

np.random.seed(100)

p = 12; T = 100
n = 4; m = 3
x_0 = 20 * np.array([1, 2, -1, -0.5]).reshape(n,1)

LQR_model = LQR(p=p, T=T, x_0=x_0)

def loss_true(K):
    return LQR_model.compute_cost(K)

def loss_noisy(K):
    return LQR_model.compute_cost_noisy(K)

def get_norm_loss_error(loss_ks, loss_0, loss_star, multi=1):
    loss_error = (np.mean(loss_ks, axis=1) - loss_star) / (loss_0 - loss_star)
    loss_error = list(np.repeat(loss_error, multi))
    loss_error.insert(0, 1)
    return loss_error

K_star = np.array([
    [1.60233232e-01, -1.36227805e-01, -9.93576677e-02, -4.28244630e-02],
    [7.47596033e-02,  9.05753832e-02,  7.46951286e-02, -1.53947620e-01],
    [3.65372978e-01, -2.59862175e-04,  5.91522023e-02, 8.25660846e-01]])

theta_star = K_star.flatten()
# loss_star = loss_true(theta_star)
loss_star = 4149.38952236

# inital value
K_0 = np.ones(K_star.shape) * 2
theta_0 = K_0.flatten()
loss_0 = loss_true(theta_0)
print("loss_0", loss_0)

# parameters
a = 0.0001; c = 0.1; A = 100
alpha = 0.602; gamma = 0.151
iter_num = 10; rep_num = 1

print("running 2SPSA")
TwoSPSA_solver = TwoSPSA(a=0.005, A=100, alpha=alpha, c=0.5, gamma=gamma, w=0.5,
                    iter_num=500, dir_num=1, rep_num=1,
                    theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy)
# TwoSPSA_solver.train()
TwoSPSA_loss_error = get_norm_loss_error(TwoSPSA_solver.loss_ks, loss_0, loss_star)
plt.figure(); plt.grid()
plt.plot(TwoSPSA_loss_error, 'k-')

print("running SPSA")
SPSA_solver = SPSA(a=a, c=c, A=A, alpha=alpha, gamma=gamma,
                    iter_num=int(iter_num/2), rep_num=rep_num,
                    theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy)
# SPSA_solver.train()
SPSA_loss_error = get_norm_loss_error(SPSA_solver.loss_ks, loss_0, loss_star)
plt.figure(); plt.grid()
plt.plot(SPSA_loss_error, 'k-')
# with open('data/LQR-SPSA-' + str(today) + '.npy', 'wb') as f:
#     np.save(f, SPSA_solver.loss_ks)

print("running CS-SPSA")
CsSPSA_solver = CsSPSA(a=a, c=c, A=A, alpha=alpha, gamma=gamma,
                       iter_num=iter_num, rep_num=rep_num,
                       theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy)
CsSPSA_solver.train()
CsSPSA_loss_error = get_norm_loss_error(CsSPSA_solver.loss_ks, loss_0, loss_star)
plt.figure(); plt.grid()
plt.plot(CsSPSA_loss_error, 'k-')
# with open('data/LQR-CsSPSA-' + str(today) + '.npy', 'wb') as f:
#     np.save(f, CsSPSA_solver.loss_ks)