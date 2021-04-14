import numpy as np

class SkewedQuartic:
    def __init__(self, p=1, noise_std=1):
        self.p = p
        self.noise_std = 1

        self.B = np.triu(np.ones((p,p))) / p
        self.theta_star = np.zeros(p)

    def loss_truth(self, theta):
        L = theta.T.dot(self.B.T.dot(self.B)).dot(theta) \
            + 0.1 * np.sum(self.B.dot(theta) ** 3) \
            + 0.01 * np.sum(self.B.dot(theta) ** 4)
        return float(L)

    def loss_noisy(self, theta):
        return self.loss_truth(theta) + np.random.normal(loc=0.0, scale=self.noise_std)

    def grad_truth(self, theta):
        grad = self.B.T.dot(
            2 * self.B.dot(theta)
            + 0.3 * np.sum(self.B.dot(theta) ** 2)
            + 0.04 * np.sum(self.B.dot(theta) ** 3))
        return grad

    def Hess_truth(self, theta):
        Hess = self.B.T.dot(
            np.diag(2 + 0.6 * self.B.dot(theta)
                    + 0.12 * np.sum(self.B.dot(theta) ** 2))).dot(self.B)
        return Hess

if __name__ == "__main__":
    p = 5
    skewed_quartic = SkewedQuartic(p)
    theta = np.ones(p)
    print(skewed_quartic.loss_truth(theta))
    print(skewed_quartic.loss_noisy(theta))
    print(skewed_quartic.grad_truth(theta))
    print(skewed_quartic.Hess_truth(theta))