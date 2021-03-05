#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 23:26:13 2020

@author: longwang
"""

import numpy as np
from algorithm.fdsa import FDSA

class CsFDSA(FDSA):
    def get_grad_est(self, iter_idx=0, rep_idx=0, theta_k=None):
        grad_k_all = np.empty((self.p, self.direct_num))

        c_k = self.c / (iter_idx + 1) ** self.gamma
        for direct_idx in range(self.direct_num):
            for i in range(self.p):
                theta_k_plus = np.array(theta_k, dtype = complex)
                theta_k_plus[i] += c_k * 1j
                loss_plus = self.loss_noisy(theta_k_plus)
                grad_k_all[i,direct_idx] = loss_plus.imag / c_k

        return np.average(grad_k_all, axis=1)