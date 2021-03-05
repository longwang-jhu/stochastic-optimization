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

        # get training and testing
        self.n_train = int(self.n * 0.8)
        self.n_test = self.n - self.n_train

        self.X_train = self.X[:self.n_train, ]
        self.y_train = self.y[:self.n_train]

        self.X_test = self.X[self.n_train:, ]
        self.y_test = self.y[self.n_train:]

    def entropy_loss(self, X, y, theta):
        return np.mean((1 - np.exp(-np.square(y - X.dot(theta)) / self.sigma2)) * self.sigma2 / 2)

    def train_loss(self, sample_idx, theta):
        return self.entropy_loss(self.X_train[sample_idx,], self.y_train[sample_idx,], theta)

    def test_loss(self, sample_idx, theta):
        return self.entropy_loss(self.X_test[sample_idx,], self.y_test[sample_idx,], theta)
