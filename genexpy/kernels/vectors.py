"""
Kernels for numerical vectors.

An experimental result is the vector of the scores of the na alternatives under one experimental condition.
A sample of N results is an array of shape (na, N), one column per condition.
"""

import numbers
import warnings

import numpy as np
import pandas as pd

from typing import Literal, Union, Tuple

from sklearn.metrics.pairwise import rbf_kernel

from genexpy.kernels import base
from genexpy.utils import rankings as ru


class VectorKernel(base.Kernel):
    """Base class of the kernels for numerical vectors. Samples have shape (na, n): one column per result."""

    def __init__(self, support: np.array = None, seed: int = 0, *args, **kwargs):
        super().__init__()
        self.support = support  # support of the results
        self.K = None  # gram matrix of the support
        self.rng = np.random.default_rng(seed)  # kept for backwards compatibility, not used

    def set_support(self, support: ru.UniverseAM):
        """Set the support and clear the cached Gram matrix."""
        self.support = support
        self.K = None

    def get_eps(self, delta, na: int = None):
        """MMD threshold epsilon corresponding to the kernel-specific similarity threshold delta."""
        raise NotImplementedError

    def _validate_parameters(self):
        pass

    @staticmethod
    def _validate_inputs(x1: np.array, x2: np.array):
        if np.shape(x1) != np.shape(x2):
            raise ValueError("Array dimensions do not match.")

    def _set_parameters(self, *args, **kwargs):
        pass

    def __call__(self, x1: np.array, x2: np.array, use_rv: bool = True) -> float:
        """
        Kernel between two vectors of scores (1-D arrays of length na), or Gram matrix between two samples
        of shape (na, n). `use_rv` is ignored, and kept for consistency with the kernels for rankings.
        """
        self._validate_inputs(x1, x2)

        return self._fun(x1, x2)

    def _fun(self, x1: np.array, x2: np.array) -> float:
        """
        The function that calls the kernel.
        """
        raise NotImplementedError

    def gram_matrix(self, s1: np.ndarray, s2: np.ndarray) -> np.ndarray[float]:
        """Gram matrix between two samples of shape (na, n1) and (na, n2); output of shape (n1, n2)."""
        raise NotImplementedError

    def __repr__(self):
        return "VectorKernel"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def get_subsample_pair(s: np.ndarray[float], subsample_size: int, disjoint: bool = True, replace: bool = False,
                           seed: Union[int, np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw two subsamples of `subsample_size` columns from `s` (shape (na, N)).

        If disjoint, the two subsamples are drawn from two disjoint halves of the columns. `seed` can be an int or a
        np.random.Generator (which is then advanced).
        """
        na, n = s.shape

        rng = np.random.default_rng(seed)

        if disjoint:
            shuffled = rng.permutation(s.T)
            out1 = rng.choice(shuffled[:n // 2], subsample_size, replace=replace).T
            out2 = rng.choice(shuffled[n // 2:], subsample_size, replace=replace).T
        else:
            out1 = rng.choice(s.T, subsample_size, replace=replace).T
            out2 = rng.choice(s.T, subsample_size, replace=replace).T

        if (out1.shape != out2.shape) or (out1.shape[0] != s.shape[0]):
            raise AssertionError(f"Wrong shape of subsamples: s has shape ({s.shape}) while the subsamples have shape "
                                 f"({out1.shape}) and ({out2.shape}).")

        return out1, out2

    def _mmd_distribution_naive(self, s: np.ndarray[float], n: int, rep: int, disjoint: bool = True,
                                replace: bool = False, seed: int = None) -> np.ndarray[float]:
        """
        MMD of `rep` pairs of subsamples of size n drawn from s, of shape (na, N) = (n_features, n_samples).
        """

        rng = np.random.default_rng(seed)  # one generator for all the pairs

        out = []
        for _ in range(rep):
            s1, s2 = self.get_subsample_pair(s, subsample_size=n, disjoint=disjoint, replace=replace, seed=rng)

            Kxx = self.gram_matrix(s1, s1)
            Kxy = self.gram_matrix(s1, s2)
            Kyy = self.gram_matrix(s2, s2)

            if Kxx.shape != (n, n) or Kxy.shape != (n, n) or Kyy.shape != (n, n):
                raise AssertionError("Wrong dimensionality for Gram matrix. Probably the sample has wrong shape: it "
                                     "should be (na, n).")

            mmd = np.sqrt(np.abs(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()))
            out.append(mmd)

        return np.array(out)

    def _mmd_distribution_embedding(self, s: np.ndarray[float], n: int, rep: int, seed: int = 0, disjoint: bool = True,
                                    replace: bool = False, use_cached_support_matrix: bool = False) -> np.ndarray[
        float]:

        raise NotImplementedError

    def _mmd_icdf_approximation(self, s: np.ndarray[float], n: int, rep: int, alpha_min: float = 0.6,
                                alpha_max: float = 1) -> np.ndarray[float]:
        """
        Close-form approximation of the quantile function (ICDF) of the MMD at `rep` equispaced levels in
        [alpha_min, alpha_max). See ``base.approximate_mmd_icdf``.
        """
        support, counts = np.unique(s, axis=1, return_counts=True)  # unique columns, (na, m)
        pmf = counts / np.sum(counts)
        K = self.gram_matrix(support, support)
        return base.approximate_mmd_icdf(K, pmf, n=n, rep=rep, alpha_min=alpha_min, alpha_max=alpha_max)

    def mmd_distribution(self, s: np.ndarray, n: int, rep: int, seed: int = 0, disjoint: bool = True,
                         replace: bool = False,
                         method: Literal["naive", "embedding", "approximation"] = "naive",
                         use_cached_support_matrix: bool = False, alpha_min=0.7, alpha_max=1) -> np.ndarray[float]:
        """
        Estimate the distribution of the MMD between two samples of n results drawn from `s` (shape (na, N)).

        Parameters
        ----------
        s : np.ndarray
            The results to draw the pairs of subsamples from, of shape (na, N).
        n, rep, seed, disjoint, replace :
            Size of the subsamples, number of pairs, random seed, and sampling scheme, as in
            ``RankingKernel.mmd_distribution``.
        method : {"naive", "approximation"}
            "naive": MMD of every pair of subsamples. "approximation": close-form approximation of the quantile
            function of the MMD at `rep` levels in [alpha_min, alpha_max) (not a sample of the MMD).
        use_cached_support_matrix : bool
            Ignored, kept for consistency with the kernels for rankings.

        Returns
        -------
        np.ndarray
            Array of shape (rep, ).
        """

        match method:
            case "naive":
                return self._mmd_distribution_naive(s, n=n, rep=rep, disjoint=disjoint, replace=replace, seed=seed)
            case "embedding":
                return self._mmd_distribution_embedding(s, n=n, rep=rep, seed=seed,
                                                        disjoint=disjoint, replace=replace,
                                                        use_cached_support_matrix=use_cached_support_matrix)
            case "approximation":
                warnings.warn("The output of calling the function with method=approximation is not a sample of the MMD"
                              " but its icdf.")
                return self._mmd_icdf_approximation(s, n=n, rep=rep, alpha_min=alpha_min, alpha_max=alpha_max)
            case _:
                raise ValueError(f"Invalid method {method}.")

    def mmd_distribution_many_n(self, s: np.ndarray, nmin: int, nmax: int, step: int,
                                seed: int = 100, disjoint: bool = True, replace: bool = False, N: int = None,
                                method: Literal["naive", "embedding", "approximation"] = "naive",
                                **mmd_distribution_parms) -> pd.DataFrame:
        """
        Run ``mmd_distribution`` for n in range(nmin, nmax, step) (seed * n is the seed for size n).
        Output columns: n, mmd, method, N, disjoint, replace, kernel.
        """
        mmds = {n: self.mmd_distribution(s=s, n=n, seed=seed * n, disjoint=disjoint, replace=replace,
                                         method=method, **mmd_distribution_parms)
                for n in range(nmin, nmax, step)}

        dfmmd = pd.DataFrame(mmds).melt(var_name="n", value_name="mmd")
        dfmmd["method"] = method
        dfmmd["N"] = N
        dfmmd["disjoint"] = disjoint
        dfmmd["replace"] = replace
        dfmmd["kernel"] = self.__str__()

        return dfmmd


class RBFKernel(VectorKernel):
    """
    Gaussian (RBF) kernel: k(x, y) = exp(-gamma * ||x - y||^2), for vectors of scores of the na alternatives.

    Goal: the results should agree on the scores of all alternatives.

    Parameters
    ----------
    gamma : float or "auto"
        Bandwidth. "auto" sets gamma = 1 / na and requires na.
    na : int, optional
        Number of alternatives.
    """

    def __init__(self, gamma: Union[float, Literal["auto"]] = "auto", na: int = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.na = na
        self.gamma = gamma
        self._set_parameters(na)
        self._validate_parameters()

    def __repr__(self):
        return f"RBFKernel(gamma={self.gamma:.2f})"

    def latex_str(self):
        return fr"$k_\text{{RBF}}^{{\gamma={self.gamma:.2f}}}$"

    def get_eps(self, delta, na: int = None):
        """
        epsilon(delta) = sqrt(2 * (1 - exp(-gamma * na * delta))), where delta is the maximum mean squared difference
        between the scores.
        """
        na = na if na is not None else self.na
        if na is None:
            raise ValueError("The number of alternatives na must be passed.")
        return np.sqrt(2 * (1 - np.exp(- self.gamma * na * delta)))

    def _validate_parameters(self):
        if isinstance(self.gamma, bool) or not isinstance(self.gamma, numbers.Real) or self.gamma < 0:
            raise ValueError(f"Invalid value for parameter gamma={self.gamma}. Accepted: positive float or 'auto'")
        self.gamma = float(self.gamma)

    def _set_parameters(self, na):
        if self.gamma == "auto":
            if na is None:
                raise ValueError("If gamma == 'auto', parameter na has to be passed.")
            self.gamma = 1 / na

    def _fun(self, x: np.ndarray[float], y: np.ndarray[float]):
        if np.ndim(x) == 1:  # two single results
            return float(rbf_kernel(np.reshape(x, (1, -1)), np.reshape(y, (1, -1)), gamma=self.gamma)[0, 0])
        return rbf_kernel(x.T, y.T, gamma=self.gamma)

    def gram_matrix(self, s1: np.ndarray[float], s2: np.ndarray[float] = None) -> np.ndarray[float]:
        """Gram matrix between two samples of shape (na, n1) and (na, n2) (s2 = s1 if None); output (n1, n2)."""
        s2 = s1 if s2 is None else s2
        return rbf_kernel(s1.T, s2.T, gamma=self.gamma)
