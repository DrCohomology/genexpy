import warnings
from itertools import product

import numpy as np
import pandas as pd

from typing import Literal, TypeAlias, Union, Tuple

from sklearn.metrics.pairwise import rbf_kernel

from genexpy.kernels import base
from genexpy.utils import rankings as ru


class VectorKernel(base.Kernel):

    def __init__(self, support: np.array = None, seed: int = 0, *args, **kwargs):
        super().__init__()
        self.support = support  # support of rankings
        self.K = None  # gram matrix of the support
        self.rng = np.random.default_rng(seed)

    def set_support(self, support: ru.UniverseAM):
        self.support = support
        self.K = None

    def get_eps(self, delta, na: int = None):
        pass

    def _validate_parameters(self):
        pass

    @staticmethod
    def _validate_inputs(x1: np.array, x2: np.array):
        if x1.shape != x2.shape:
            raise ValueError("Array dimensions do not match.")

    def _set_parameters(self, *args, **kwargs):
        pass

    def __call__(self, x1: np.array, x2: np.array, use_rv: bool = True) -> float:
        """
        Computes the Mallows kernel between two rankings, which is based on the difference in their rankings adjusted by a
        kernel bandwidth parameter gamma.

        Parameters:
        - x1 (Ranking): The first ranking as a RankVector or RankByte.
        - x2 (Ranking): The second ranking as a RankVector or RankByte.
        - gamma (float, 'auto'): The decay parameter for the kernel. If 'auto', it adjusts based on the length of the rankings.
        - use_rv (bool): Determines whether to use the rank vector or byte representation for the calculation.

        Returns:
        - float: The computed Mallows kernel value.

        Raises:
        - ValueError: If the rankings do not have the same gammamber of alternatives.
        """
        self._validate_inputs(x1, x2)

        return self._fun(x1, x2)

    def _fun(self, x1: np.array, x2: np.array) -> float:
        """
        The function that calls the kernel.
        """
        pass

    def gram_matrix(self, s1: np.ndarray[float], s2: np.ndarray[float]) -> np.ndarray[float]:
        raise NotImplementedError

    def __repr__(self):
        return "VectorKernel"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def get_subsample_pair(s: np.ndarray[float], subsample_size: int, disjoint: bool = True, replace: bool = False,
                           seed: int = None) -> Tuple[np.ndarray[float], np.ndarray[float]]:
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

    def _mmd_distribution_naive(self, s: np.ndarray[float], n: int, rep: int, disjoint: bool = True, replace: bool = False, seed: int = None) -> np.ndarray[float]:

        """
        s has size (na, n) = ((n_features, n_samples)
        """

        rng = np.random.default_rng(seed)

        out = []
        for _ in range(rep):
            s1, s2 = self.get_subsample_pair(s, subsample_size=n, disjoint=disjoint, replace=replace, seed=seed)

            Kxx = self.gram_matrix(s1, s1)
            Kxy = self.gram_matrix(s1, s2)
            Kyy = self.gram_matrix(s2, s2)

            if Kxx.shape != (n, n) or Kxy.shape != (n, n) or Kyy.shape != (n, n):
                raise AssertionError("Wrong dimensionality for Gram matrix. Probably the sample has wrong shape: it should be (na, n).")

            mmd = np.sqrt(np.abs(np.mean(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())))
            out.append(mmd)

        return np.array(out)

    def _mmd_distribution_embedding(self, s: np.ndarray[float], n: int, rep: int, seed: int = 0, disjoint: bool = True,
                                    replace: bool = False, use_cached_support_matrix: bool = False) -> np.ndarray[
        float]:

        raise NotImplementedError

    def _mmd_icdf_approximation(self, s: np.ndarray[float], n: int, rep: int, alpha_min: float = 0.6,
                                alpha_max: float = 1) -> np.ndarray[float]:
        """
        Iterated approximations of the CDF of the MMD.
            1. MMD^2 = sum of chi squares (from asymptotic behavior of the MMD^2)
            2. sum of chi squares = chi square (from moment matching)
            3. chi square = normal (Wilson-Hilferty method)
            4. approximate the normal CDF and ICDF

        The approximation is trustworthy for values of alpha between alpha_min = 0.6 and alpha_max < 1

        Returns
        -------

        """

        if alpha_min < 0.6:
            warnings.warn("The approximation of the MMD might not be reliable for alpha_min < 0.6.")
        if alpha_max > 1:
            raise ValueError("The maximum value of alpha_max is 1.")

        support, counts = np.unique(tuple(tuple(x) for x in s), axis=1, return_counts=True)
        pmf = counts / np.sum(counts)
        K = self.gram_matrix(support, support)

        m = len(support)
        C = np.eye(m) - 1 / m * np.ones((m, m))
        H = C @ K @ C

        Th = H @ np.diag(pmf)

        lam = np.linalg.eigvalsh(Th)

        L1 = np.sum(lam)
        L2 = np.sum(lam ** 2)
        L3 = np.sum(np.triu(np.outer(lam, lam), 1))
        L4 = 3 * L2 + 2 * L3

        # constant and dof of chi square, see Solomon et Stephens (1977)
        # r = 1  # here just for consistency with the source, where they do not fix it
        a = (L4 - L1 ** 2) / L1
        k = 2 * L1 ** 2 / (L4 - L1 ** 2)

        def normal2chisq(z: np.array, k: float, a: float):
            """
            Inverse of the Wilson-Hilferty (WH) approximation of a chi square with a normal.
            WH(a*chisq(k)) = normal(0, 1)
            WH_inv(normal(0, 1), a, k) = a*chisq(k)
            """
            return (np.sqrt(2 / (9 * k)) * z + (1 - 2 / (9 * k))) ** 3 * a * k

        def normal_ICDF(alpha: np.array):
            """
            ICDF of the normal(0, 1) from Lin (1989)'s CDF approximation
            """
            return -0.861779 + 0.00120192 * np.sqrt(514089 - 1.664 * 10 ** 6 * np.log(2 * (1 - alpha)))

        alpha = np.linspace(alpha_min, alpha_max, rep, endpoint=False)
        return np.sqrt(normal2chisq(normal_ICDF(alpha), k, a)) / np.sqrt(n)

    def mmd_distribution(self, s: ru.SampleAM, n: int, rep: int, seed: int = 0, disjoint: bool = True,
                         replace: bool = False,
                         method: Literal["naive", "embedding", "approximation"] = "naive",
                         use_cached_support_matrix: bool = False, alpha_min=0.7, alpha_max=1) -> np.ndarray[float]:

        match method:
            case "naive":
                return self._mmd_distribution_naive(s, n, rep)
            case "embedding":
                return self._mmd_distribution_embedding(s, n=n, rep=rep, seed=seed,
                                                        disjoint=disjoint, replace=replace,
                                                        use_cached_support_matrix=use_cached_support_matrix)
            case "approximation":
                warnings.warn("The output of calling the function with method=approximation is not a sample of the MMD"
                              "but its icdf.")
                return self._mmd_icdf_approximation(s, n=n, rep=rep, alpha_min=alpha_min, alpha_max=alpha_max)
            case _:
                raise ValueError(f"Invalid method {method}.")

    def mmd_distribution_many_n(self, s: ru.SampleAM, nmin: int, nmax: int, step: int,
                                seed: int = 100, disjoint: bool = True, replace: bool = False, N: int = None,
                                method: Literal["naive", "embedding", "approximation"] = "naive",
                                **mmd_distribution_parms) -> pd.DataFrame:
        mmds = {n: self.mmd_distribution(s=s, n=n, seed=int(np.exp(seed)) * n, disjoint=disjoint, replace=replace,
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

    def __init__(self, gamma: Union[float, Literal["auto"]] = "auto", na: int = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gamma = gamma
        self._set_parameters(na)
        self._validate_parameters()

    def __repr__(self):
        return f"RBFKernel(gamma={self.gamma:.2f})"

    def latex_str(self):
        return fr"$\kappa_\text{{RBF}}^{{\gamma={self.gamma:.2f}}}$"

    def get_eps(self, delta, na: int = None):
        if self.gamma == "auto":
            return np.sqrt(2 * (1 - np.exp(-delta)))
        else:
            # Use gamma / gamma_auto
            # delta is the MSE
            return np.sqrt(2 * (1 - np.exp(- self.gamma * na * delta)))

    def _validate_parameters(self):
        if isinstance(self.gamma, str):
            if self.gamma != "auto":
                raise ValueError(f"Invalid value for parameter gamma={self.gamma}. Accepted: positive float or 'auto'")
        elif isinstance(self.gamma, float):
            if self.gamma < 0:
                raise ValueError(f"Invalid value for parameter gamma={self.gamma}. Accepted: positive float or 'auto'")
        else:
            raise ValueError(f"Invalid value for parameter gamma={self.gamma}. Accepted: positive float or 'auto'")

    def _set_parameters(self, na):
        if self.gamma == "auto":
            if na is None:
                raise ValueError("If gamma == 'auto', parameter na has to be passed.")
            self.gamma = 1 / na

    def _fun(self, x: np.ndarray[float], y: np.ndarray[float]) -> float:
        return rbf_kernel(x.T, y.T, gamma=self.gamma)

    def gram_matrix(self, s1: np.ndarray[float], s2: np.ndarray[float]) -> np.ndarray[float]:
        return rbf_kernel(s1.T, s2.T, gamma=self.gamma)
