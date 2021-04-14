#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 23:26:13 2020

@author: Long Wang
"""

import numpy as np
from algorithms.fdsa import FDSA

class CsFDSA(FDSA):
    def get_grad_est(self, iter_idx=0, rep_idx=0, theta_k=None):
        c_k = self.c / (iter_idx + 1) ** self.gamma
        grad_k = np.zeros(self.p)
        for dir_idx in range(self.dir_num):
            for i in range(self.p):
                theta_k_plus = np.array(theta_k, dtype = complex)
                theta_k_plus[i] += c_k * 1j
                loss_plus = self.loss_noisy(theta_k_plus)
                grad_k[i] += loss_plus.imag / c_k
        return grad_k / self.dir_num