"""
Copyright (c) 2020 ground0state. All rights reserved.
License: MIT License
"""
import sys
import warnings

import numpy as np
import scipy as sp
from tqdm import tqdm


class DirectLearningSparseChangesDual():
    """Direct Learning of Sparse Changes
    in Markov Networks by Density Ratio Estimation.

    Parameters
    ----------
    lambda1 : float
        L2 penalty.
    lambda2 : float
        Additional L2 penalty.
    lr: float
        Learning rate.
    max_iter: int
        Max number of iterations over the train dataset
        to perform training.
    tol: float
        Tolerance for termination.
        A lower bound on the change in the value
        of the objective function during a step.
    """

    def __init__(self, lambda1, lambda2, lr=1e-8,
                 max_iter=1000, tol=1e-10):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.loss = []
        self.pmatrix_diff = None

        self._g = None
        self._H = None
        self._alpha = None

    def fit(self, X_normal, X_error):
        """Fit the DirectLearningSparseChanges model
        according to the given train data.

        Parameters
        ----------
        X_normal : array-like, shape (n_samples, n_features)
            Normal measured vectors, where n_samples
            is the number of samples
            and n_features is the number of features.

        X_error: array-like, shape (n_samples, n_features)
            Error measured vectors, where n_samples
            is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
        """
        # initialize
        self._g = -1/2*np.cov(X_normal, rowvar=False)
        self._H = -1/2*np.asarray([np.outer(x, x) for x in X_error]).T

        # initialize alpha
        temp = np.ones(X_error.shape[0])
        self._alpha = temp/temp.sum()

        self.loss = []
        immediate_loss_prev = sys.float_info.max
        for _ in tqdm(range(self.max_iter)):
            temp = self._alpha \
                - self.lr*self._obj_deri_func(self._alpha)
            temp = np.maximum(0, temp)
            self._alpha = temp/(np.sum(temp))

            immediate_loss = self._obj_func(self._alpha)
            self.loss.append(immediate_loss)
            if immediate_loss_prev - immediate_loss <= self.tol:
                break
            immediate_loss_prev = immediate_loss
        else:
            warnings.warn(
                "Objective did notconverge. The max_iter was reached.")

        theta = self._trans_dual(self._alpha)
        self.pmatrix_diff = self._theta2pmatrix(theta)

        return self

    def get_sparse_changes(self):
        """Gettter for sparse changes.

        Returns
        -------
        theta : The difference of precision matrix.
        """
        return self.pmatrix_diff

    def _obj_func(self, alpha):
        xi = self._g - self._H@alpha

        obj = np.sum(sp.special.xlogy(alpha, alpha))
        temp = (np.maximum(np.abs(xi)-self.lambda2, 0))**2
        temp = np.triu(temp)
        temp = np.sum(temp)
        obj += 1/(2*self.lambda1)*temp
        return obj

    def _obj_deri_func(self, alpha):
        xi = self._g - self._H@alpha

        # calculate gamma
        gamma = np.zeros(xi.shape)
        for i in range(xi.shape[0]):
            for j in range(xi.shape[1]):
                if xi[i, j] > self.lambda2:
                    gamma[i, j] = -(xi[i, j]-self.lambda2)/self.lambda1
                elif xi[i, j] < -self.lambda2:
                    gamma[i, j] = -(xi[i, j]+self.lambda2)/self.lambda1
                else:
                    gamma[i, j] = 0

        # calculate derivative
        dobj = np.log(
            np.where(alpha == 0, sys.float_info.min, alpha)) + 1

        temp = np.zeros(self._H.shape[2])
        for i in range(gamma.shape[0]):
            for j in range(gamma.shape[1]):
                if j < i:
                    continue
                temp += gamma[i, j]*self._H[i, j, :]
        dobj += temp
        return dobj

    def _trans_dual(self, alpha):
        xi = self._g - self._H@alpha

        theta = np.zeros(xi.shape)
        it = np.nditer(xi, flags=['multi_index'])
        while not it.finished:
            norm = np.abs(it[0])
            if norm > self.lambda2:
                theta[it.multi_index] = 1/self.lambda1 * \
                    (1-self.lambda2/norm)*it[0]
            else:
                theta[it.multi_index] = 0

            it.iternext()

        return theta

    def _theta2pmatrix(self, theta):
        diag = np.diag(np.diag(theta))
        return (theta + diag)/2


if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler

    mu_n = [0, 0, 0]
    sigma_n = [
        [0.75, -0.5,  0.25],
        [-0.5,  1., -0.5],
        [0.25, -0.5,  0.75]]
    X_normal = np.random.multivariate_normal(
        [0, 0, 0], sigma_n, 10000)

    mu_e = [0, 0, 0]
    sigma_e = [
        [0.75,  0.25, -0.5],
        [0.25,  0.75, -0.5],
        [-0.5, -0.5,  1.]]
    X_error = np.random.multivariate_normal(
        [0, 0, 0], sigma_e, 10000)

    labels = ["col0", "col1", "col2"]

    sc = StandardScaler()
    X1 = sc.fit_transform(X_normal)
    X2 = sc.fit_transform(X_error)

    model = DirectLearningSparseChangesDual(
        lambda1=0.1, lambda2=0.3, max_iter=10000)
    model.fit(X1, X2)
    pmatrix_diff = model.get_sparse_changes()

    # 学習経過
    plt.plot(model.loss)
    plt.show()

    # 可視化
    plt.figure(figsize=(5, 5))
    sns.heatmap(pmatrix_diff, xticklabels=labels,
                yticklabels=labels, cmap="coolwarm", center=0,
                annot=True, square=True, fmt='.2f')
    plt.show()
