"""
This code is from public python library pyrfm
What follows was copied verbatim from https://neonnnnn.github.io/pyrfm/_modules/pyrfm/random_feature/random_fourier.html#RandomFourier
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from math import sqrt
import warnings


class RandomFourier(BaseEstimator, TransformerMixin):
    """Approximates feature map of the RBF kernel by
    Random Fourier Feature map.

    Parameters
    ----------
    n_components : int (default=100)
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.

    kernel : str (default="rbf")
        Kernel to be approximated.
        Now only "rbf" can be used.

    gamma : float or str (default="auto")
        Parameter for the RBF kernel.

    use_offset : bool (default=False)
        If True, Z(x) = (cos(w_1x+b_1), cos(w_2x+b_2), ... , cos(w_Dx+b_D),
        where w is random_weights and b is offset (D=n_components).
        If False, Z(x) = (cos(w_1x), ..., cos(w_{D/2}x), sin(w_1x), ...,
        sin(w_{D/2}x)).

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If np.RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    random_weights_ : array, shape (n_features, n_components) (use_offset=True) or (n_features, n_components/2) (otherwise)
        The sampled basis.

    random_offset_ : array or None, shape (n_components, )
        The sampled offset vector. If use_offset=False, random_offset_=None.

    References
    ----------
    [1] Random Features for Large-Scale Kernel Machines.
    Ali Rahimi and Ben Recht.
    In NIPS 2007.
    (https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)

    """

    def __init__(
        self,
        n_components=100,
        kernel="rbf",
        gamma="auto",
        use_offset=False,
        random_state=None,
    ):
        self.n_components = n_components
        self.gamma = gamma
        self.kernel = kernel
        self.use_offset = use_offset
        self.random_state = random_state

    def fit(self, X, y=None):
        """Generate random weights according to n_features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the transformer.
        """
        random_state = check_random_state(self.random_state)
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape

        if self.use_offset:
            n_components = self.n_components
        else:
            n_components = int(self.n_components / 2)
            if self.n_components % 2 != 0:
                self.n_components = 2 * n_components
                warnings.warn(
                    "n_components % 2 != 0. n_components is changed "
                    " to {}.".format(self.n_components)
                )

        if self.gamma == "auto":
            gamma = 1.0 / X.shape[1]
        else:
            gamma = self.gamma

        size = (n_features, n_components)
        # TODO: Implement other shift-invariant kernels
        if self.kernel in ["rbf", "gaussian"]:
            self.random_weights_ = random_state.normal(size=size)
            self.random_weights_ *= sqrt(2 * gamma)
        else:
            raise ValueError(
                "Kernel {} is not supported. "
                'Use "rbf" or "Gaussian"'.format(self.kernel)
            )
        if self.use_offset:
            self.random_offset_ = random_state.uniform(
                0, 2 * np.pi, size=self.n_components
            )
        else:
            self.random_offset_ = None

        return self

    def transform(self, X):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self, "random_weights_")
        X = check_array(X, accept_sparse=True)
        output = safe_sparse_dot(X, self.random_weights_)
        if self.use_offset:
            output += self.random_offset_
            output = np.cos(output)
        else:
            output = np.hstack((np.cos(output), np.sin(output)))
        return sqrt(2.0 / self.n_components) * output

    def _remove_bases(self, indices):
        if not self.use_offset:
            warnings.warn("Bases are not removed when use_offset=False.")
            return False
        else:
            self.random_weights_ = np.delete(self.random_weights_, indices, 1)
            self.random_offset_ = np.delete(self.random_offset_, indices, 0)
            self.n_components = len(self.random_offset_)
            return True
