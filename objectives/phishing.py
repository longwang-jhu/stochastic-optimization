#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 20:07:13 2021

@author: Long Wang
"""
import numpy as np
from sklearn.datasets import load_svmlight_file

class Phishing:
    def __init__(self):
        data = load_svmlight_file('data/phishing')
        X, y = data[0].toarray(), data[1]
        y[y == 0] = -1

        # shuffle data
        np.random.seed(0)
        perm_idx = np.random.permutation(len(X))

        self.X = X[perm_idx] # shape = (11055,68)
        self.y = y[perm_idx] # shape = (11055,)

        self.n, self.p = X.shape
        self.sigma2 = 100

    def batch_loss(self, sample_idx, theta):
        X_sample = self.X[sample_idx,]
        y_sample = self.y[sample_idx,]

        result = np.mean((1 - np.exp(-np.square(y_sample - X_sample.dot(theta)) / self.sigma2)) * self.sigma2 / 2)
        return result