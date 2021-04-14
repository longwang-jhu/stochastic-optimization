#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 23:41:54 2020

@author: Long Wang
"""
import numpy as np

class GradDescent(object):
    def __init__(self, a=0.1, A=0, alpha=0.602,
                 iter_num=100, dir_num=1, rep_num=1,
                 theta_0=None, loss_obj=None,
                 record_theta=True, record_loss=True,
                 seed=0):

        # step size: a_k = a / (k + 1 + A) ** alpha
        # iter_num: number of iterations
        # dir_num: number of random directions per iteration
        # rep_num: number of replicate

        self.a = a
        self.A = A
        self.alpha = alpha

        self.iter_num = iter_num
        self.dir_num = dir_num
        self.rep_num = rep_num

        self.theta_0 = theta_0
        self.loss_obj = loss_obj

        self.record_theta = record_theta
        self.record_loss = record_loss

        np.random.seed(seed)

        self.p = theta_0.shape[0]
        if self.record_theta:
            self.theta_ks = np.zeros((self.p, self.iter_num, self.rep_num))
        if self.record_loss:
            self.loss_ks = np.zeros((self.iter_num, self.rep_num))

    def train(self):
        pass

    def record(self, iter_idx=0, rep_idx=0, theta_k=None):
        if self.record_theta:
            self.theta_ks[:,iter_idx,rep_idx] = theta_k
        if self.record_loss:
            self.loss_ks[iter_idx,rep_idx] = self.loss_obj.loss_truth(theta_k)

    def show(self, iter_idx, rep_idx, iter_every):
        if self.record_loss and divmod(iter_idx + 1, iter_every)[1] == 0:
            print("iter:", iter_idx + 1, "loss:", self.loss_ks[iter_idx,rep_idx])