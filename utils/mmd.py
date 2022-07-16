import numpy as np
from sklearn import metrics


class MMD:
    def __init__(self):
        super().__init__()

    def Liner(self, X, Y):
        delta = X.mean(0) - Y.mean(0)
        return delta.dot(delta.T)

    def Rbf(self, X, Y, gamma=1.0):
        XX = metrics.pairwise.rbf_kernel(X, Y, gamma)
        YY = metrics.pairwise.rbf_kernel(X, Y, gamma)
        XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
        return XX.mean() + YY.mean() - 2 * XY.mean()

    def Poly(self, X, Y, degree=2, gamma=1.0, coef=0):
        XX = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef)
        YY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef)
        XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef)
        return XX.mean() + YY.mean() - 2 * XY.mean()


if __name__ == '__main__':
    a = [[1, 0.3, 2, 0.5], [2, 0.1, 4, 0.2], [4, 0.43, 1, 0.5]]
    b = [[2, 0.3, 1, 0.5], [1, 0.1, 5, 0.2], [3, 0.23, 2, 1.5]]
    mmd = MMD()
    print(mmd.Liner(np.array(a), np.array(b)))
    print(mmd.Rbf(np.array(a), np.array(b)))
    print(mmd.Poly(np.array(a), np.array(b)))