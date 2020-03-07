"""
Copyright (c) 2019-2020 ground0state. All rights reserved.
License: MIT License
"""
import sys
import warnings

import numpy as np


class KLDensityRatioEstimation():
    """Kullback-Leibler density ratio estimation.

    Parameters
    ----------
    band_width : float
        Smoothing parameter gaussian kernel.
    lr: float
        Learning rate.
    max_iter: int
        Max number of iterations over the train dataset
        to perform training.
    """

    def __init__(self, band_width=1.0, lr=0.1, max_iter=100, tol=1e-4):
        self.band_width = band_width
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter

        self.theta = None
        self.X_center = None
        self.psi = None
        self.psi_prime = None
        # losses of objective function in training
        self.loss = []

    def fit(self, X_normal, X_error):
        """Fit the DensityRatioEstimation model
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

        Notes
        -----
        Use X_normal for basic function.
        """

        # prepare basic function
        self.X_center = X_normal
        self.psi = np.asarray([self._gaussian_kernel(x, X_normal)
                               for x in X_normal])
        self.psi_prime = np.asarray(
            [self._gaussian_kernel(x, X_normal) for x in X_error])

        # initialize theta
        self.theta = np.ones(len(self.psi)) / len(self.psi)

        # initilalize density latio
        r = np.dot(self.psi, self.theta)
        r_prime = np.dot(self.psi_prime, self.theta)

        # execute gradient method
        self.loss = []
        immediate_loss_prev = sys.float_info.max

        for _ in range(self.max_iter):
            # update theta
            temp = self.theta - self.lr * self._obj_deri_func(r)
            self.theta = np.maximum(temp, 0)

            # calculate density latio
            r = np.dot(self.psi, self.theta)
            r_prime = np.dot(self.psi_prime, self.theta)

            # calculate loss
            immediate_loss = self._obj_func(r, r_prime)
            self.loss.append(immediate_loss)

            if immediate_loss_prev - immediate_loss <= self.tol:
                break
            immediate_loss_prev = immediate_loss
        else:
            warnings.warn(
                "Objective did notconverge. The max_iter was reached.")

        return self

    def _obj_func(self, r, r_prime):
        obj = np.sum(r_prime)/len(self.psi_prime) \
            - np.sum(np.log(r))/len(self.psi)
        return obj

    def _obj_deri_func(self, r):
        dobj = self.psi_prime.sum(axis=0) / len(self.psi_prime) - \
            (self.psi / r).sum(axis=0) / len(self.psi)
        return dobj

    def _gaussian_kernel(self, x, X):
        psi = np.exp(-np.sum((x - X)**2, axis=1)/(2*self.band_width**2))
        return psi

    def oof_score(self, X_normal, X_error):
        """Calculate objective according to the given oof data.

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
        obj : array-like, shape (n_samples,)
            Objective.
        """
        psi = np.asarray([self._gaussian_kernel(x, self.X_center)
                          for x in X_normal])
        psi_prime = np.asarray(
            [self._gaussian_kernel(x, self.X_center) for x in X_error])

        r = np.dot(psi, self.theta)
        r_prime = np.dot(psi_prime, self.theta)

        obj = np.sum(r_prime)/len(psi_prime) \
            - np.sum(np.log(r))/len(psi)
        return obj

    def score(self, X_error):
        """Calculate anomaly score according to the given test data.

        Parameters
        ----------
        X_error: array-like, shape (n_samples, n_features)
            Error measured vectors, where n_samples
            is the number of samples
            and n_features is the number of features.

        Returns
        -------
        anomaly_score : array-like, shape (n_samples,)
            Anomaly score.
        """

        psi_prime = np.asarray([self._gaussian_kernel(x, self.X_center)
                                for x in X_error])
        r_prime = np.dot(psi_prime, self.theta)
        anomaly_score = -np.log(r_prime)
        return anomaly_score


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    from sklearn.model_selection import KFold

    def generate_data():
        mu1 = np.array([0, 0, 0])
        sigma1_inv = np.array([
            [1, 0,  1],
            [0, 1,  -1],
            [1, -1,  3]])
        sigma1 = np.linalg.inv(sigma1_inv)
        # output
        # sigma1 = np.array([[2., -1., -1.],
        #                    [-1.,  2.,  1.],
        #                    [-1.,  1.,  1.]])
        x1 = stats.multivariate_normal(mu1, sigma1).rvs(size=200)

        mu2 = np.array([0, 0])
        sigma2_inv = np.array([
            [1, 0.9],
            [0.9, 1, ]])
        sigma2 = np.linalg.inv(sigma2_inv)
        # output
        # sigma2 = np.array([[5.26315789, -4.73684211],
        #                    [-4.73684211,  5.26315789]])
        x2 = stats.multivariate_normal(mu2, sigma2).rvs(size=200)

        X = np.concatenate([x1, x2], axis=1)

        X_normal = X[0:100]
        X_error = X[100:200]

        X_error[9:14, 0] = 7
        X_error[29:34, 1] = -5
        X_error[49:54, 2] = 6
        X_error[63:68, 3] = 5
        X_error[81:86, 4] = -6

        return X_normal, X_error

    # prepare data
    np.random.seed(0)
    X_normal, X_error = generate_data()

    # cross validation for band width
    hs = np.arange(1.0, 4.0, 0.1)
    hs_score = {}
    for h in hs:
        losses = []
        for train_index, valid_index in KFold(n_splits=3).split(X_normal):
            train_X_normal = X_normal[train_index]
            valid_X_normal = X_normal[valid_index]
            train_X_error = X_error[train_index]
            valid_X_error = X_error[valid_index]

            model = KLDensityRatioEstimation(
                band_width=h, lr=0.001, max_iter=100000)
            model.fit(train_X_normal, train_X_error)
            losses.append(model.oof_score(
                valid_X_normal, valid_X_error))

        hs_score[h] = np.mean(losses)

    # select best band width
    min_h = min(hs_score, key=hs_score.get)
    print(min_h)

    # prediction model
    model = KLDensityRatioEstimation(
        band_width=min_h, lr=0.001, max_iter=100000)
    model.fit(X_normal, X_error)

    # bandwidth cross validation plot
    plt.plot([float(s)
              for s in list(hs_score.keys())], list(hs_score.values()))
    plt.xlabel("Band width")
    plt.ylabel("objective")
    plt.show()

    # learning curve
    plt.plot(model.loss)
    plt.show()

    # anomaly score
    a = model.score(X_error)
    plt.plot(a)
    plt.show()

    # KLdivergence
    a_ = -np.mean(model.score(X_normal))
    print(a_)
