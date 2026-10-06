"""
Base class for kernels, and helpers shared by all kernels.

Kernels are looked up by class name, which is how the configuration files and the
precomputed-MMD file names refer to them (e.g., ``"MallowsKernel(nu=0.00202)"``).
The helpers at the bottom of the module implement the close-form approximation of the
quantile function (ICDF) of the MMD used by every kernel family.
"""

import re
import warnings

import numpy as np

from genexpy.kernels import utils


def find_subclass(superclass, subclass_name: str):
    """Return the (direct or indirect) subclass of `superclass` called `subclass_name`."""
    for subclass in utils.all_subclasses(superclass):
        if subclass.__name__ == subclass_name:
            return subclass
    raise ValueError(f"No subclass named '{subclass_name}' found for {superclass.__name__}")


class Kernel:
    """
    Base class of all kernels.

    A kernel ``k(x, y)`` measures the similarity of two experimental results. Subclasses implement
    the kernel (``__call__``), its Gram matrix (``gram_matrix``), the estimation of the distribution
    of the MMD (``mmd_distribution``, ``mmd_distribution_many_n``), and the map from the
    user-facing similarity threshold delta to the MMD threshold epsilon (``get_eps``).
    """

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def _find_subclass(cls, subclass_name: str):
        return find_subclass(cls, subclass_name)

    @classmethod
    def from_string(cls, s: str) -> "Kernel":
        """
        Instantiate a kernel from its string representation, e.g., ``Kernel.from_string("MallowsKernel(nu=0.04)")``.

        The parameters must be explicit numbers: ``"MallowsKernel(nu='auto')"`` does not work, as 'auto'
        needs the number of alternatives. Note that ``str(kernel)`` rounds the parameters (e.g., to five
        decimals for ``nu``), so the parsed kernel can differ slightly from the one that produced the string.
        """
        match = re.fullmatch(r"(\w+)\((.*)\)", s.strip())
        if not match:
            raise ValueError(f"Invalid kernel string: {s}")

        class_name, args_str = match.groups()
        target_class = cls._find_subclass(class_name)

        kwargs = {}
        if args_str:
            try:
                # Only allow a restricted environment for safety
                kwargs = eval(f"dict({args_str})", {"__builtins__": {}, "dict": dict})
            except Exception as e:
                raise ValueError(f"Invalid argument list '{args_str}': {e}")

        return target_class(**kwargs)

    @classmethod
    def from_name_and_parameters(cls, kernel_name: str, **kernel_args) -> "Kernel":
        """
        Instantiate a kernel from its class name and parameters.

        Example: ``Kernel.from_name_and_parameters("MallowsKernel", nu="auto", na=10)``.
        """
        kernel_cls = cls._find_subclass(kernel_name)
        return kernel_cls(**kernel_args)

    def mmd_distribution_many_n(self, sample, nmin, nmax, step, rep, disjoint, replace, method, N,
                                use_cached_support_matrix):
        """Estimated distribution of the MMD for several subsample sizes n (see the subclasses)."""
        raise NotImplementedError

    def get_eps(self, delta, na):
        """MMD threshold epsilon corresponding to the kernel-specific similarity threshold delta."""
        raise NotImplementedError

    def latex_str(self) -> str:
        """LaTeX name of the kernel, with its parameters (used in the plots)."""
        return r"$k$"


# ----------------------------------------------------------------------------------------------------------------------
# Close-form approximation of the quantile function of the MMD
# ----------------------------------------------------------------------------------------------------------------------

def mmd_spectrum_moments(K: np.ndarray, pmf: np.ndarray) -> tuple[float, float]:
    """
    First two moments of the spectrum of the centered kernel operator of a discrete distribution.

    For two independent samples of size n from P, n * MMD_n^2 converges in distribution to
    sum_i 2 * lambda_i * chi2_1, where lambda_i are the eigenvalues of the kernel centered w.r.t. P,
    i.e., of M = D^{1/2} (K - K p 1^T - 1 p^T K + p^T K p) D^{1/2}, with D = diag(p).
    M is symmetric, so L1 = sum(lambda) = trace(M) and L2 = sum(lambda^2) = ||M||_F^2:
    no eigendecomposition is needed.

    Parameters
    ----------
    K : np.ndarray
        Gram matrix of the support of P, shape (m, m).
    pmf : np.ndarray
        Probability of every element of the support, shape (m, ).

    Returns
    -------
    L1, L2 : float
        Sum of the eigenvalues and sum of their squares.
    """
    K = np.asarray(K, dtype=float)
    pmf = np.asarray(pmf, dtype=float)
    Kp = K @ pmf
    Kc = K - Kp[:, None] - Kp[None, :] + pmf @ Kp
    sqp = np.sqrt(pmf)
    M = sqp[:, None] * Kc * sqp[None, :]
    return float(np.trace(M)), float(np.sum(M * M))


def chi2_moment_matching(L1: float, L2: float) -> tuple[float, float]:
    """
    Scale `a` and degrees of freedom `k` of the a * chi2(k) variable with the same mean (2 L1) and
    variance (8 L2) as sum_i 2 * lambda_i * chi2_1 (Solomon and Stephens, 1977).

    Algebraically identical to a = (L4 - L1^2) / L1 and k = 2 L1^2 / (L4 - L1^2), with L4 = L1^2 + 2 L2.
    """
    return 2 * L2 / L1, L1 ** 2 / L2


def normal_icdf_lin(alpha):
    """Quantile function of the standard normal, from Lin (1989)'s approximation of its CDF."""
    return -0.861779 + 0.00120192 * np.sqrt(514089 - 1.664 * 10 ** 6 * np.log(2 * (1 - alpha)))


def approximate_mmd_icdf(K: np.ndarray, pmf: np.ndarray, n: int, rep: int, alpha_min: float = 0.6,
                         alpha_max: float = 1) -> np.ndarray:
    """
    Close-form approximation of the quantile function of MMD_n at `rep` equispaced levels in [alpha_min, alpha_max).

    Iterated approximations:
        1. n * MMD_n^2 ~ sum of chi-squares (asymptotic distribution of the MMD, Gretton et al. 2012);
        2. sum of chi-squares ~ a * chi2(k) (moment matching, Solomon and Stephens 1977);
        3. a * chi2(k) ~ normal (Wilson-Hilferty);
        4. normal quantile function from Lin (1989).
    The approximation is trustworthy for alpha_min >= 0.6 and alpha_max < 1.
    """
    if alpha_min < 0.6:
        warnings.warn("The approximation of the MMD might not be accurate for alpha_min < 0.6.")
    if alpha_max > 1:
        raise ValueError("The maximum value of alpha_max is 1.")

    L1, L2 = mmd_spectrum_moments(K, pmf)
    if np.isclose(L1, 0.0):  # degenerate distribution: the MMD is always 0
        return np.zeros(rep)
    a, k = chi2_moment_matching(L1, L2)
    z = normal_icdf_lin(np.linspace(alpha_min, alpha_max, rep, endpoint=False))
    # inverse of the Wilson-Hilferty transform: a * chi2(k) from a standard normal
    chisq = (np.sqrt(2 / (9 * k)) * z + (1 - 2 / (9 * k))) ** 3 * a * k
    return np.sqrt(chisq) / np.sqrt(n)
