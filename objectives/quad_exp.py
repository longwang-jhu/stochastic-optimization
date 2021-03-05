#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:57:59 2020

@author: Long Wang
"""

import numpy as np
from scipy.optimize import minimize

class QuadExp:
    def __init__(self, eta=None):
        self.eta = eta
        self.p = eta.shape[0]

    def loss_true(self, theta=None):
        return np.sum(theta ** 2) + np.sum(self.eta / (self.eta + theta))

    def loss_noisy(self, theta=None):
        X = np.random.exponential(1 / self.eta)
        return np.sum(theta ** 2) + np.sum(np.exp(-(X * theta)))

    def grad_true(self, theta=None):
        return 2 * theta - self.eta / (self.eta + theta) ** 2

    def get_theta_star(self):
        self.theta_star = minimize(self.loss_true, np.ones(self.p), method='Nelder-Mead', tol=1e-6).x

if __name__ == "__main__":
    # np.random.seed(10)
    p = 10
    eta = np.array([1.1025, 1.6945, 1.4789, 1.9262, 0.7505,
                1.3267, 0.8428, 0.7247, 0.7693, 1.3986])
    quad_exp_model = QuadExp(eta)
    quad_exp_model.get_theta_star()
    print(quad_exp_model.theta_star)

    theta_k = np.ones(p) * 2
    print(quad_exp_model.loss_true(theta_k))
    print(quad_exp_model.loss_noisy(theta_k))