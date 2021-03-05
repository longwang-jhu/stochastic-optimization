#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:57:59 2020

@author: longwang
"""

import numpy as np

def loss_true(eta, theta):
    return np.sum(theta ** 2) + np.sum(eta / (eta + theta))

def loss_noisy(X, theta):
    # theta: parameter value, shape = (p,)
    return np.sum(theta ** 2) + np.sum(np.exp(-(X * theta)))

def grad_true(eta, theta):
    return 2 * theta - eta / (eta + theta) ** 2

if __name__ == "__main__":
    # np.random.seed(10)
    p = 10
    theta_0 = np.ones(p) * 2
    loss_value = loss_noisy(theta_0)
    loss_value_true = loss_true(theta_0)
    print(loss_value, loss_value_true)

    from scipy.optimize import minimize
    res = minimize(loss_true, theta_0, method='Nelder-Mead', tol=1e-6)
    print(res.x)

    # theta_complex = np.ones(p, dtype = complex)
    # theta_complex[1] = 1+1j
    # loss_value_complex = loss_noisy(theta_complex)
    # print(loss_value_complex)