#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:31:45 2020

@author: Long Wang
"""

import numpy as np
from algorithms.opti_algo import OptiAlgo

class SPSABatch(OptiAlgo):
    def __init__(self, a=0, A=0, alpha=0.602, c=0, gamma=0.101,
             iter_num=1, dir_num=1, rep_num=1,
             theta_0=None, loss_true=None, loss_noisy=None,
             record_theta_flag=True, record_loss_flag=True,
             n_batch=0, n_mini_batch=0):

        super(SPSABatch, self).__init__(
            a, A, alpha, c, gamma,
            iter_num, dir_num, rep_num,
            theta_0, loss_true, loss_noisy,
            record_theta_flag, record_loss_flag)

        self.n_batch = n_batch
        self.n_mini_batch = n_mini_batch

    def get_delta_all(self):
        self.delta_all = np.round(np.random.rand(self.p, self.dir_num, self.iter_num, self.rep_num)) * 2 - 1

    def get_grad_est(self, iter_idx=0, rep_idx=0, theta_k=None):
        c_k = self.c / (iter_idx + 1) ** self.gamma
        grad_k = np.zeros(self.p)
        for dir_idx in range(self.dir_num):
            sample_idx = np.random.choice(self.n_batch, self.n_mini_batch, replace=False)
            delta_k = self.delta_all[:, dir_idx, iter_idx, rep_idx]
            loss_plus = self.loss_noisy(sample_idx, theta_k + c_k * delta_k)
            loss_minus = self.loss_noisy(sample_idx, theta_k - c_k * delta_k)
            grad_k += (loss_plus - loss_minus) / (2 * c_k) * delta_k
        return grad_k / self.dir_num

    def train(self):
        self.get_delta_all()
        for rep_idx in range(self.rep_num):
            print("running rep_idx:", rep_idx+1, "/", self.rep_num)
            theta_k = self.theta_0.copy() # reset theta_k
            for iter_idx in range(self.iter_num):
                a_k = self.a / (iter_idx + 1 + self.A) ** self.alpha
                g_k = self.get_grad_est(iter_idx, rep_idx, theta_k)
                theta_k -= a_k * g_k

                # record result
                self.record_result(iter_idx, rep_idx, theta_k)
                # show result
                self.show_result(iter_idx, rep_idx)