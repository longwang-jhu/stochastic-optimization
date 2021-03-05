#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 4 14:01:37 2021

@author: Long Wang
"""

import numpy as np
from scipy import linalg
from algorithms.opti_algo import OptiAlgo

class TwoSPSA(OptiAlgo):
    def __init__(self, a=0, A=0, alpha=0.602, c=0, gamma=0.101, w=0,
                 iter_num=1, dir_num=1, rep_num=1,
                 theta_0=None, loss_true=None, loss_noisy=None,
                 record_theta_flag=False, record_loss_flag=True):

        super(TwoSPSA, self).__init__(a, A, alpha, c, gamma,
                                      iter_num, dir_num, rep_num,
                                      theta_0, loss_true, loss_noisy,
                                      record_theta_flag, record_loss_flag)
        self.w = w
        self.H_k = np.eye(self.p)

    def get_delta_all(self):
        self.delta_all = np.round(np.random.rand(
            self.p, self.dir_num, self.iter_num, self.rep_num)) * 2 - 1

    def get_deltat_all(self):
        self.deltat_all = np.round(np.random.rand(
            self.p, self.dir_num, self.iter_num, self.rep_num)) * 2 - 1

    def get_grad_Hhat_est(self, iter_idx=0, rep_idx=0, theta_k=None):
        c_k = self.c / (iter_idx + 1) ** self.gamma
        ct_k = c_k

        grad_k = np.zeros(self.p)
        Hhat_k = np.zeros((self.p, self.p))
        for dir_idx in range(self.dir_num):
            delta_k = self.delta_all[:, dir_idx, iter_idx, rep_idx]
            loss_plus = self.loss_noisy(theta_k + c_k * delta_k)
            loss_minus = self.loss_noisy(theta_k - c_k * delta_k)
            grad_k += (loss_plus - loss_minus) / (2 * c_k) * delta_k

            deltat_k = self.deltat_all[:, dir_idx, iter_idx, rep_idx]
            losst_plus = self.loss_noisy(theta_k + c_k * delta_k + c_k * deltat_k)
            losst_minus = self.loss_noisy(theta_k - c_k * delta_k + c_k * deltat_k)
            loss_diff = ((losst_plus - loss_plus) - (losst_minus - loss_minus)) / (2 * c_k * ct_k)
            Hhat_k += loss_diff * delta_k.reshape(self.p,1).dot(deltat_k.reshape(1, self.p))
        grad_k /= self.dir_num
        Hhat_k /= self.dir_num
        Hhat_k = (Hhat_k + Hhat_k.T) / 2 # make it symmetric
        return grad_k, Hhat_k

    def update_H_est(self, iter_idx=0, Hhat_k=None):
        w_k = self.w / (iter_idx + 2)
        Hbar_k = (1 - w_k) * self.H_k + w_k * Hhat_k

        Hbar_k_eig, Hbar_k_vec = linalg.eigh(Hbar_k)
        Hbar_k_eig = np.maximum(1e-6, np.absolute(Hbar_k_eig)) # make it PD
        Hbarbar_k = Hbar_k_vec.dot(np.diag(Hbar_k_eig)).dot(Hbar_k_vec.T)
        self.H_k = Hbarbar_k

    def train(self):
        self.get_delta_all()
        self.get_deltat_all()

        for rep_idx in range(self.rep_num):
            print("running rep_idx:", rep_idx+1, "/", self.rep_num)
            theta_k = self.theta_0.copy() # reset theta_k
            for iter_idx in range(self.iter_num):
                a_k = self.a / (iter_idx + 1 + self.A) ** self.alpha
                grad_k, Hhat_k = self.get_grad_Hhat_est(iter_idx, rep_idx, theta_k)
                self.update_H_est(iter_idx, Hhat_k)
                theta_k -= a_k * linalg.solve(self.H_k, grad_k)

                # record result
                self.record_result(iter_idx, rep_idx, theta_k)

                # show result
                self.show_result(iter_idx, rep_idx)