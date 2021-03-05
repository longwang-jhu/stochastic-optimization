#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 19:23:01 2021

@author: Long Wang
"""
from datetime import date
import numpy as np
import matplotlib.pyplot as plt

from algorithms.spsa_batch import SPSABatch
from algorithms.two_spsa_batch import TwoSPSABatch

from objectives.phishing import Phishing

###
today = date.today()
np.random.seed(100)

# construct model
Phishing_model = Phishing()
n = Phishing_model.n
p = Phishing_model.p

n_train = Phishing_model.n_train
n_test = Phishing_model.n_test

def loss_noisy(sample_idx, theta):
    return Phishing_model.train_loss(sample_idx, theta)

def loss_true(theta):
    return Phishing_model.test_loss(range(n_test), theta)

# start algorithm
theta_0 = np.ones(p)
loss_0 = loss_true(theta_0)

# parameters
a = 0.01; c = 0.2; A = 100
alpha = 0.602; gamma = 0.151
iter_num = 5000; rep_num = 1

print("running SPSA")
SPSA_solver = SPSABatch(
    a=a, c=c, A=A, alpha=alpha, gamma=gamma,
    iter_num=iter_num, rep_num=rep_num,
    theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy,
    record_theta_flag=False,
    n_batch=n_train, n_mini_batch=100)

SPSA_solver.train()
SPSA_loss_ks = np.mean(SPSA_solver.loss_ks, axis=1)
plt.figure(); plt.grid()
plt.plot(SPSA_loss_ks)

print("running 2SPSA")
twoSPSA_solver = TwoSPSABatch(
    a=0.005, A=100, alpha=alpha, c=0.5, gamma=gamma, w=0.5,
    iter_num=5000, dir_num=1, rep_num=1,
    theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy,
    record_theta_flag=False,
    n_batch=n_train, n_mini_batch=100)

twoSPSA_solver.train()
twoSPSA_loss_ks = np.mean(twoSPSA_solver.loss_ks, axis=1)
plt.figure(); plt.grid()
plt.plot(twoSPSA_loss_ks)