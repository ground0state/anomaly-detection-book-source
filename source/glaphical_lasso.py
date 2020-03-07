"""
Copyright (c) 2019-2020 ground0state. All rights reserved.
License: MIT License
"""
import numpy as np
from tqdm import tqdm


class GraphicalLasso():
    """Fit the model according to the train data.

    Parameters
    ----------
    rho: float
        Inverse of the scale. The larger this is,
        precision matrix elements become sparse.
    max_iter: int
        Max iteration.
    max_iter_beta: int
        Max iteration of graphical lasso.
    tol: float
        When the update amount of the precision matrix
        becomes smaller than this tolerance value,
        the coordinate descent stops.
    tol_beta: float
        When the update amount of beta
        becomes smaller than this tolerance,
        the graphical lasso stops.
    """

    def __init__(self, rho=0.01, max_iter=10, max_iter_beta=50,
                 tol=None, tol_beta=None):
        self.rho = rho
        self.max_iter = max_iter
        self.max_iter_beta = max_iter_beta
        self.tol = tol
        self.tol_beta = tol_beta

        self.cov = None
        self.pmatrix = None
        self.pmatrix_inv = None
        self.n_features = None

    def fit(self, X):
        """Fit the model according to the train data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Normal measured vectors,
            where n_samples is the number of samples.

        Returns
        -------
        self : object
        """
        # feature dimension
        self.n_features = X.shape[1]

        # feature dimension must be larger than 2
        if self.n_features <= 1:
            raise ValueError("Feature size must be >=2")

        # solve precision matrix
        self.pmatrix, self.pmatrix_inv, self.cov \
            = self._solve(X, rho=self.rho,
                          max_iter=self.max_iter,
                          max_iter_beta=self.max_iter_beta,
                          tol=self.tol,
                          tol_beta=self.tol_beta)
        return self

    def _solve(self, X, rho,
               max_iter, max_iter_beta,
               tol=None, tol_beta=None):
        # covariance
        cov = np.cov(X, rowvar=False, bias=False)

        # initialize tolerance
        if tol is None:
            tol = 0.001*np.abs(np.diag(cov)).mean()
        if tol_beta is None:
            tol_beta = 0.0001

        # initialize precision matrix and inverse of precision matrix
        pmatrix = np.ones((X.shape[1], X.shape[1]))
        pmatrix_inv = cov + rho*np.diag(np.ones(X.shape[1]))

        # Coordinate Descent
        pmatrix, pmatrix_inv = self._coordinate_descent(
            cov, pmatrix, pmatrix_inv, rho,
            max_iter, max_iter_beta, tol, tol_beta)

        return pmatrix, pmatrix_inv, cov

    def _coordinate_descent(self, cov, pmatrix, pmatrix_inv, rho,
                            max_iter, max_iter_beta, tol, tol_beta):
        pmatrix_new = np.copy(pmatrix)
        pmatrix_inv_new = np.copy(pmatrix_inv)
        for _ in tqdm(range(max_iter), total=max_iter):
            for i in range(len(cov)):
                W = np.delete(np.delete(pmatrix_inv, i, 0), i, 1)
                s = np.delete(cov[:, i], i, axis=0)

                # graphical lasso
                beta = self._glasso(W, s, rho,
                                    max_iter_beta, tol_beta)

                # update pmatrix_inv
                w = beta@W
                sigma = cov[i, i] + rho
                w_ = np.insert(w, i, sigma)
                pmatrix_inv[:, i] = w_
                pmatrix_inv[i, :] = w_

                # update pmatrix
                lam = 1 / (sigma - beta@W@beta)
                l = - lam * beta
                l_ = np.insert(l, i, lam)
                pmatrix[:, i] = l_
                pmatrix[i, :] = l_

            if np.abs(
                    pmatrix_inv_new - pmatrix_inv).mean() <= tol:
                return pmatrix_new, pmatrix_inv_new

            pmatrix = pmatrix_new
            pmatrix_inv = pmatrix_inv_new

        return pmatrix, pmatrix_inv

    def _glasso(self, W, s, rho, max_iter_beta, tol_beta):
        W_offdiag = W - np.diagflat(np.diag(W))
        beta = np.zeros(W.shape[0])
        beta_new = np.copy(beta)
        for _ in range(max_iter_beta):
            A = s - beta@W_offdiag
            for idx, a in enumerate(A):
                beta_new[idx] = np.sign(
                    a) * np.maximum(np.abs(a) - rho, 0.0)/W[idx, idx]

            if np.abs(beta_new - beta).mean() <= tol_beta:
                return beta_new
            beta = beta_new
        return beta

    def score(self, X):
        """Calculate anomaly score
        according to the given test data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Error measured vectors, where n_samples
            is the number of samples
            and n_features is the number of features.

        Returns
        -------
        anomaly_score : array-like, shape (n_samples, n_features)
            Anomaly score.
        """
        # check feature size
        if self.n_features != X.shape[1]:
            raise ValueError("Feature size must be"
                             "same as training data")

        # calculate anomaly score
        diag = np.diag(self.pmatrix)
        anomaly_score = []
        for x in X:
            a = np.log(2*np.pi/diag)/2 \
                + (x@self.pmatrix)**2/(2*diag)
            anomaly_score.append(a)

        return np.array(anomaly_score)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pydotplus
    import seaborn as sns
    from PIL import Image
    from scipy import stats
    from sklearn.preprocessing import StandardScaler

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

    # generate data
    np.random.seed(0)
    X_normal, X_error = generate_data()
    cols = ["col0", "col1", "col2", "col3", "col4"]

    # check data
    plt.plot(X_normal)
    plt.legend(cols)
    plt.show()

    # check data
    plt.plot(X_error)
    plt.legend(cols)
    plt.show()

    # standardize
    sc = StandardScaler()
    X1 = sc.fit_transform(X_normal)
    X2 = sc.fit_transform(X_error)

    # graphical lasso
    model = GraphicalLasso(rho=0.3)
    model.fit(X1)

    # plot anomaly score
    a = model.score(X2)
    plt.plot(a)
    plt.legend(cols)
    plt.show()

    # markov graph
    def save_markov_graph(pmatrix, cols, r=2.0, cut_off=0.1,
                          file_name='default.png', show_plot=False):

        # set layout
        num_nodes = len(cols)
        pos = {}
        for i in range(num_nodes):
            theta = - 2 * np.pi * i / num_nodes + np.pi / 2
            pos[i] = r * \
                np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)

        graph = pydotplus.Dot(graph_type='graph')

        # add nodes
        for i, c in enumerate(cols):
            node = pydotplus.Node(c, pos="{},{}!".format(*pos[i]))
            graph.add_node(node)

        # add edges
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if (j > i) and (np.abs(pmatrix[i, j]) > cut_off):
                    val = pmatrix[i, j]

                    edge = pydotplus.Edge(graph.get_node(c1)[0],
                                          graph.get_node(c2)[0],
                                          penwidth=10*np.abs(val))
                    edge.set_label('{:.2f}'.format(val))
                    h, s, v = 0.0 if val > 0 else 2 / \
                        3, np.abs(val), 1.0  # 色を設定する
                    edge.set_color(' '.join([str(a) for a in (h, s, v)]))
                    graph.add_edge(edge)

        # plot graph
        graph.set_layout('neato')
        graph.write_png(file_name, prog='dot')

        if show_plot:
            img = Image.open(file_name)
            img.show()

    def partial_corr(pmatrix):
        h = np.diag(np.power(np.diag(pmatrix), -0.5))
        pcorr = - h@pmatrix@h
        pcorr += np.eye(pcorr.shape[0])
        return pcorr

    # partial correlation
    pcorr = partial_corr(model.pmatrix)
    sns.heatmap(pcorr, cmap="coolwarm", center=0,
                annot=True, square=True, fmt='.2f')
    plt.show()

    # pairwise Markov graph
    save_markov_graph(pcorr, cols=cols, show_plot=True)

    # anomaly analysis
    model1 = GraphicalLasso(rho=0.3)
    model1.fit(X1)
    model2 = GraphicalLasso(rho=0.3)
    model2.fit(X2)

    def anomaly_analysis(cov1, pmatrix1, cov2, pmatrix2):
        pmatrix1_diag = np.diag(pmatrix1)
        pmatrix2_diag = np.diag(pmatrix2)
        a = 1/2*np.log(pmatrix1_diag/pmatrix2_diag) - 1 / \
            2*(np.diag(pmatrix1@cov1@pmatrix1)/pmatrix1_diag -
               np.diag(pmatrix2@cov1@pmatrix2)/pmatrix2_diag)
        b = 1/2*np.log(pmatrix2_diag/pmatrix1_diag) - 1 / \
            2*(np.diag(pmatrix2@cov2@pmatrix2)/pmatrix2_diag -
               np.diag(pmatrix1@cov2@pmatrix1)/pmatrix1_diag)
        return np.maximum(a, b)

    a = anomaly_analysis(model1.cov, model1.pmatrix,
                         model2.cov, model2.pmatrix)
    plt.bar(cols, a)
    plt.show()
