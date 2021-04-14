import numpy as np
from algos.grad_descent import GradDescent

class MSPSA(GradDescent):
    def __init__(self, d=0, c=0.1, gamma=0.101, **kwargs):
        self.d = d # the first d components are discrete
        self.c = c
        self.gamma = gamma
        super().__init__(**kwargs)

    def pi(self, theta=None):
        # project the first d components to floor(x) + 0.5
        pi_theta = theta.copy()
        pi_theta[:self.d] = np.floor(pi_theta[:self.d]) + 0.5
        return pi_theta

    def project(self, theta=None):
        # project the first d components to the nearest integer
        proj_theta = theta.copy()
        proj_theta[:self.d] = np.round(proj_theta[:self.d])
        return proj_theta

    def get_delta_ks(self):
        self.delta_ks = np.round(np.random.rand(self.p, self.dir_num, self.iter_num, self.rep_num)) * 2 - 1

    def get_grad_est(self, iter_idx=0, rep_idx=0, theta_k=None):
        c_k = self.c / (iter_idx + 1) ** self.gamma
        C_k = np.hstack((np.repeat(0.5, self.d), np.repeat(c_k, self.p - self.d)))
        grad_k = np.zeros(self.p)
        for dir_idx in range(self.dir_num):
            delta_k = self.delta_ks[:, dir_idx, iter_idx, rep_idx]
            loss_plus = self.loss_obj.loss_noisy(self.pi(theta_k) + C_k * delta_k)
            loss_minus = self.loss_obj.loss_noisy(self.pi(theta_k) - C_k * delta_k)
            grad_k += (loss_plus - loss_minus) / (2 * C_k * delta_k)
        return grad_k / self.dir_num

    def train(self):
        self.get_delta_ks()
        for rep_idx in range(self.rep_num):
            print("running rep_idx:", rep_idx + 1, "/", self.rep_num)
            theta_k = self.theta_0.copy()
            for iter_idx in range(self.iter_num):
                a_k = self.a / (iter_idx + 1 + self.A) ** self.alpha
                g_k = self.get_grad_est(iter_idx, rep_idx, theta_k)
                theta_k -= a_k * g_k

                # record result
                if self.record_theta:
                    self.theta_ks[:,iter_idx,rep_idx] = self.project(theta_k)
                if self.record_loss:
                    self.loss_ks[iter_idx,rep_idx] = self.loss_obj.loss_truth(self.project(theta_k))