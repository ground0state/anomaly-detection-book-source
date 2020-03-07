"""
Copyright (c) 2020 ground0state. All rights reserved.
License: MIT License
"""
import sys
import warnings

import numpy as np
from tqdm import tqdm


class DirectLearningSparseChanges():
    """Direct Learning of Sparse Changes
    in Markov Networks by Density Ratio Estimation.

    Parameters
    ----------
    lambda1 : float
        L2 penalty.
    lambda2 : float
        L1 penalty.
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

    def __init__(self, lambda1, lambda2,
                 lr=0.01,  max_iter=1000,
                 tol=1e-4):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.loss = []
        self.theta = None

        self._S = None
        self._G = None

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
        self._S = np.cov(X_normal, rowvar=False)
        self._G = -1/2*np.asarray([np.outer(x, x) for x in X_error]).T

        # initialize theta
        self.theta = self._S

        self.loss = []
        immediate_loss_prev = sys.float_info.max
        for _ in tqdm(range(self.max_iter)):
            self.theta = self._prox(self.theta - self.lr *
                                    self._obj_deri_func(self.theta),
                                    self.lr, self.lambda2)

            immediate_loss = self._obj_func(self.theta)
            self.loss.append(immediate_loss)
            if immediate_loss_prev - immediate_loss <= self.tol:
                break
            immediate_loss_prev = immediate_loss
        else:
            warnings.warn(
                "Objective did notconverge. The max_iter was reached.")
        return self

    def _prox(self, v, eta, lam):
        return np.sign(v) * np.maximum(np.abs(v) - eta*lam, 0.0)

    def get_sparse_changes(self):
        """Gettter for sparse changes.

        Returns
        -------
        theta : The difference of precision matrix.
        """
        return self.theta

    def _obj_func(self, theta):
        temp = np.zeros(self._G.shape[2])
        for i in range(self._G.shape[0]):
            for j in range(self._G.shape[1]):
                temp = self._G[i, j, :]*theta[j, i]

        obj = 1/2*np.trace(theta@self._S) \
            + np.log(np.mean(np.exp(temp))) \
            + 1/2*self.lambda1*np.sum(theta**2) \
            + self.lambda2*np.sum(np.abs(theta))
        return obj

    def _obj_deri_func(self, theta):
        dobj = self._S/2

        temp = np.zeros(self._G.shape[2])
        for i in range(self._G.shape[0]):
            for j in range(self._G.shape[1]):
                temp = self._G[i, j, :]*theta[j, i]

        dobj += self._G@np.exp(temp)/np.sum(np.exp(temp))
        dobj += self.lambda1 * theta

        return dobj


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

    model = DirectLearningSparseChanges(
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
