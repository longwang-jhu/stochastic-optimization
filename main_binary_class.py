#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 19:23:01 2021

@author: Long Wang
"""
from datetime import date
import numpy as np
import random
import matplotlib.pyplot as plt

from algorithms.spsa import SPSA
from algorithms.two_spsa import TwoSPSA

from objectives.phishing import Phishing

###
today = date.today()
np.random.seed(100)

# construct model
Phishing_model = Phishing()
n = Phishing_model.n
p = Phishing_model.p

def loss_noisy(theta):
    sample_idx = random.sample(range(n), 100)
    return Phishing_model.batch_loss(sample_idx, theta)

def loss_true(theta):
    return Phishing_model.batch_loss(range(n), theta)

# start algorithm
theta_0 = np.ones(p)
loss_0 = loss_true(theta_0)

# parameters
a = 0.01; c = 0.2; A = 100
alpha = 0.602; gamma = 0.151
iter_num = 5000; rep_num = 1

print("running SPSA")
SPSA_solver = SPSA(a=a, c=c, A=A, alpha=alpha, gamma=gamma,
                   iter_num=iter_num, rep_num=rep_num,
                   theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy,
                   record_theta_flag=False)
# SPSA_solver.train()
# loss_ks = np.mean(SPSA_solver.loss_ks, axis=1)
# plt.figure(); plt.grid()
# plt.plot(loss_ks)

print("running 2SPSA")
TwoSPSA_solver = TwoSPSA(a=0.005, A=100, alpha=alpha, c=0.5, gamma=gamma, w=0.5,
                    iter_num=5000, dir_num=1, rep_num=1,
                    theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy,
                    record_theta_flag=False)
TwoSPSA_solver.train()
TwoSPSA_loss_error = np.mean(TwoSPSA_solver.loss_ks, axis=1)
plt.figure(); plt.grid()
plt.plot(TwoSPSA_loss_error, 'k-')






