"""
Utility module with probability distributions over rankings.

Every distribution samples rankings of `na` alternatives (or from a given support) and returns them as a
``genexpy.utils.rankings.SampleAM``. Distributions keep their own random generator: successive calls to
``sample`` return different (but reproducible, given the seed) samples.

- ``UniformDistribution``: uniform over all rankings (with or without ties).
- ``DegenerateDistribution`` / ``MDegenerateDistribution``: concentrated on one / m rankings.
- ``SpikeDistribution``: rankings close (w.r.t. a kernel) to a center are more likely.
- ``PMFDistribution``: arbitrary pmf over a support, e.g., the empirical distribution of a sample.
"""
import math
import numpy as np
import time

from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache
from numbers import Integral
from scipy.special import factorial, stirling2
from typing import Literal, Union

from .kernels import rankings as ku
from .utils import rankings as ru


def get_unique_ranks_distribution(n, exact=False, normalized=True):
    r"""
    Calculates the distribution of the number of unique ranks in uniformly random rankings of length 'n'.

    A ranking with ties has a different number of unique ranks. For instance,
    0112 has 3 unique ranks. This function computes the probability of
    observing a ranking with 'k' unique ranks for all 1 <= k <= n.

    Parameters
    ----------
    n : int
        The length of the rankings.
    exact : bool, optional
        Whether to use exact calculations or approximations. The default is False.
    normalized : bool, optional
        Whether to normalize the distribution. The default is True.

    Returns
    -------
    np.ndarray
        A 1D array containing the probabilities of observing each number of
        unique ranks in rankings of length 'n'.

    Notes
    -----
    The number of rankings of n alternatives with k unique ranks is :math:`k! S(n, k)`, where :math:`S(n, k)` is
    the Stirling number of the second kind, i.e., the number of ways to partition a set of 'n' elements into 'k'
    non-empty subsets. See also the terms n(n-1)/2 + 1 to n(n+1)/2 of T(n, k) in https://oeis.org/A019538.
    """
    out = factorial(np.arange(n)+1, exact=exact) * stirling2(n, np.arange(n)+1, exact=exact)
    out = out.astype(float)
    return out / out.sum() if normalized else out


@lru_cache(maxsize=None)
def _ordered_partitions_table(n: int) -> tuple:
    """
    T[m][k] = k! S(m, k), the number of rankings of m alternatives with exactly k tiers, as exact integers,
    from T(m, k) = k * (T(m-1, k-1) + T(m-1, k)).
    """
    T = [[0] * (n + 1) for _ in range(n + 1)]
    T[0][0] = 1
    for m in range(1, n + 1):
        for k in range(1, m + 1):
            T[m][k] = k * (T[m - 1][k - 1] + T[m - 1][k])
    return tuple(tuple(row) for row in T)


def _to_ranking_bytes(x):
    """Encode a ranking given as bytes, AdjacencyMatrix (2-D) or rank vector (1-D) as the bytes of its adjacency
    matrix. None is returned unchanged."""
    if x is None or isinstance(x, bytes):
        return None if x is None else bytes(x)
    x = np.asarray(x)
    if x.ndim == 2:
        return ru.AdjacencyMatrix(x).tohashable()
    if x.ndim == 1:
        return ru.AdjacencyMatrix.from_rank_vector(x).tohashable()
    raise ValueError(f"Cannot interpret an object of shape {x.shape} as a ranking.")


def _repeat_rankings(elements: list, reps: int) -> ru.SampleAM:
    """SampleAM holding `elements` (bytes) tiled `reps` times."""
    out = np.empty(len(elements) * reps, dtype=object)
    out[:] = list(elements) * reps
    return out.view(ru.SampleAM)


class FunctionDefaultDict(defaultdict):
    """
    A defaultdict subclass that initializes values using a function.

    This class extends defaultdict to automatically create missing values by
    calling a specified function.

    Parameters
    ----------
    func : callable
        The function to use for initializing missing values.
    *args :
        Arguments to pass to the defaultdict constructor.
    **kwargs :
        Keyword arguments to pass to the defaultdict constructor.
    """
    def __init__(self, func, *args, **kwargs):
        super().__init__(func, *args, **kwargs)
        self.func = func

    def __missing__(self, key):
        """
        Called when a missing key is accessed.

        Parameters
        ----------
        key : any
            The missing key.

        Returns
        -------
        any
            The value returned by the function for the missing key.
        """
        return self.func(key)


class ProbabilityDistribution(ABC):
    """
    Abstract base class for probability distributions over rankings.

    This class defines the common interface for probability distributions
    over rankings, including methods for sampling, calculating probabilities,
    and accessing distribution properties.

    Parameters
    ----------
    support : ru.SampleAM, optional
        The support of possible rankings. If None, the support is defined by
        the number of alternatives (na). The default is None.
    na : int, optional
        The number of alternatives in the ranking. Required if support is None.
        The default is None.
    ties : bool, optional
        Whether ties are allowed in the rankings. The default is True.
    seed : int, optional
        The random seed for sampling. The default is None.

    Attributes
    ----------
    support : ru.SampleAM
        The support of possible rankings.
    na : int
        The number of alternatives in the ranking.
    pmf : defaultdict
        The probability mass function of the distribution.
    ties : bool
        Whether ties are allowed in the rankings.
    seed : int
        The random seed for sampling.
    rng : np.random.Generator
        The random number generator.
    sample_time : float
        The time taken for the last sampling operation.
    name : str
        The name of the distribution.

    Methods
    -------
    sample(n: int, **kwargs) -> ru.SampleAM
        Samples 'n' rankings from the distribution.
    multisample(n: int, nm: int, **kwargs) -> ru.MultiSampleAM
        Samples 'nm' samples of 'n' rankings.
    """

    def __init__(self, support: ru.SampleAM = None, na: int = None, ties: bool = True, seed: int = None):
        self.support = support
        if support is None:
            if na is None:
                raise ValueError("Specify the number of alternatives or a support")
            self.na = na
        else:
            if len(self.support) == 0:
                raise ValueError("The input list is empty.")
            self.na = support.get_na()

        self.pmf = defaultdict(lambda: 0)  # TODO: refactor as pd.Series (better for multisamples)
        self.ties = ties
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sample_time = np.nan
        self.name = "Generic"

    def _check_valid_element(self, x):
        """
        Checks if an element (bytes of an adjacency matrix) is valid for the distribution.

        Parameters
        ----------
        x : bytes
            The element to check.

        Raises
        ------
        ValueError
            If the element is not valid for the distribution.
        """
        if x is not None:
            if self.support is not None and x not in self.support:
                raise ValueError("The input element must belong to the support.")
            else:
                if len(x) != self.na ** 2:  # bytestring representation:
                    raise ValueError("The input element must have the correct number of alternatives.")
                # TODO: check if ties are present

    def _sample_from_support(self, n: int, **kwargs):
        """
        Samples rankings uniformly, with replacement, from the support.

        Parameters
        ----------
        n : int
            The number of rankings to sample.

        Returns
        -------
        ru.SampleAM
            A sample of rankings from the support.
        """
        return self.support.get_subsample(subsample_size=n, seed=self.rng, use_key=False, replace=True)

    @abstractmethod
    def _sample_from_na(self, n: int, **kwargs) -> ru.SampleAM:
        """
        Samples rankings from the distribution based on the number of alternatives.

        Parameters
        ----------
        n : int
            The number of rankings to sample.

        Returns
        -------
        ru.SampleAM
            A sample of rankings from the distribution.
        """
        pass

    def _sample_from_na_noties(self, n: int, **kwargs) -> ru.SampleAM:
        """
        Samples rankings from the distribution without ties.

        This method should be implemented by subclasses that support sampling
        without ties.

        Parameters
        ----------
        n : int
            The number of rankings to sample.

        Returns
        -------
        ru.SampleAM
            A sample of rankings from the distribution without ties.
        """
        raise NotImplementedError

    def sample(self, n: int, **kwargs) -> ru.SampleAM:
        """
        Samples 'n' rankings from the distribution.

        Parameters
        ----------
        n : int
            The number of rankings to sample.

        Returns
        -------
        ru.SampleAM
            A sample of rankings from the distribution.
        """
        start_time = time.time()
        if self.support is not None:
            out = self._sample_from_support(n, **kwargs)
        else:
            if self.ties:
                out = self._sample_from_na(n, **kwargs)
            else:
                out = self._sample_from_na_noties(n, **kwargs)
        self.sample_time = time.time() - start_time
        return out

    def multisample(self, n: int, nm: int, **kwargs) -> ru.MultiSampleAM:
        """
        Samples 'nm' samples of 'n' rankings from the distribution.

        Parameters
        ----------
        n : int
            Size of the samples.
        nm : int
            Number of samples.

        Returns
        -------
        ru.MultiSampleAM
            Array of shape (nm, n).
        """

        return ru.MultiSampleAM([self.sample(n, **kwargs) for _ in range(nm)])

    def __str__(self):
        """Returns a string representation of the distribution."""
        return f"{self.name}(na={self.na}, ties={self.ties})"


class UniformDistribution(ProbabilityDistribution):
    """
    Uniform distribution over rankings.

    With ties=True (default), every ranking with ties of `na` alternatives (weak order) is equally likely;
    with ties=False, every permutation is. If a support is given, its elements are sampled uniformly.

    Parameters
    ----------
    support : ru.SampleAM, optional
        If given, sample uniformly from it.
    na : int, optional
        Number of alternatives, required if support is None.
    ties : bool
        Whether ties are allowed.
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> distr = UniformDistribution(na=5, seed=42)
    >>> sample = distr.sample(100)        # ru.SampleAM of 100 rankings
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Uniform"
        self._block_sizes = {}  # cache of the distributions of the size of the top tier

    def _top_tier_size_distribution(self, m: int, k: int):
        """Sizes j and probabilities of the top tier of a uniformly random ranking of m alternatives with k tiers:
        P(j) = binom(m, j) T(m - j, k - 1) / T(m, k)."""
        if (m, k) not in self._block_sizes:
            T = _ordered_partitions_table(self.na)
            js = np.arange(1, m - k + 2)
            p = np.array([math.comb(m, int(j)) * T[m - j][k - 1] / T[m][k] for j in js])
            self._block_sizes[(m, k)] = (js, p / p.sum())
        return self._block_sizes[(m, k)]

    def _sample_from_na(self, n: int, **kwargs) -> ru.SampleAM:
        """
        Samples rankings uniformly among all rankings with ties of na alternatives.

        1. The number of tiers k is drawn from its distribution (see get_unique_ranks_distribution).
        2. The sizes of the tiers are drawn sequentially, from the best tier down, from their exact conditional
           distribution, which makes every ranking with k tiers equally likely.
        3. The alternatives are randomly assigned to the tiers.

        Parameters
        ----------
        n : int
            The number of rankings to sample.

        Returns
        -------
        ru.SampleAM
            A sample of rankings from the uniform distribution.
        """
        nurs = self.rng.choice(np.arange(self.na) + 1, p=get_unique_ranks_distribution(self.na), size=n)  # number of unique ranks
        rf = np.empty((n, self.na), dtype=int)
        for i, nur in enumerate(nurs):
            sizes = []
            m = self.na
            for k in range(nur, 0, -1):  # tiers left to fill
                js, p = self._top_tier_size_distribution(m, k)
                j = int(self.rng.choice(js, p=p)) if len(js) > 1 else int(js[0])
                sizes.append(j)
                m -= j
            rf[i, self.rng.permutation(self.na)] = np.repeat(np.arange(nur), sizes)
        return ru.SampleAM.from_rank_vector_matrix(rf.T)

    def _sample_from_na_noties(self, n: int, **kwargs) -> ru.SampleAM :
        """
        Samples rankings from the uniform distribution without ties.

        Parameters
        ----------
        n : int
            The number of rankings to sample.

        Returns
        -------
        ru.SampleAM
            A sample of rankings from the uniform distribution without ties.
        """
        return ru.SampleAM.from_rank_vector_matrix(
            self.rng.permuted(np.tile(np.arange(self.na), n).reshape(n, self.na), axis=1).T)

    def latex_str(self):
        return rf"$U_{{{self.na}}}$"


class DegenerateDistribution(ProbabilityDistribution):
    """
    Degenerate distribution concentrated on a single ranking.

    This class represents a degenerate distribution where all probability mass
    is concentrated on a single ranking.

    Parameters
    ----------
    *args :
        Arguments passed to the ProbabilityDistribution constructor.
    **kwargs :
        Keyword arguments passed to the ProbabilityDistribution constructor.
    element : bytes, ru.AdjacencyMatrix, or rank vector, optional
        The ranking on which the distribution is concentrated. If None, it is
        sampled from the uniform distribution at the first call of `sample`, and kept afterwards.
        The default is None.
    """
    def __init__(self, *args, element: ru.AdjacencyMatrix = None, **kwargs):
        super().__init__(*args, **kwargs)
        element = _to_ranking_bytes(element)
        self._check_valid_element(element)
        self._uniform = UniformDistribution(self.support, self.na, ties=self.ties, seed=self.seed)
        self.element = element
        self.name = "Degenerate"

    def _get_element(self) -> bytes:
        if self.element is None:
            self.element = self._uniform.sample(1)[0]
        return self.element

    def _sample_from_support(self, n: int, **kwargs):
        return _repeat_rankings([self._get_element()], n)

    def _sample_from_na(self, n: int, **kwargs):
        return _repeat_rankings([self._get_element()], n)

    _sample_from_na_noties = _sample_from_na


class MDegenerateDistribution(ProbabilityDistribution):
    """
    Multi-degenerate distribution concentrated on multiple rankings.

    This class represents a distribution where all probability mass is
    concentrated (uniformly) on a set of 'm' rankings. Samples contain every element the same number of times.

    Parameters
    ----------
    *args :
        Arguments passed to the ProbabilityDistribution constructor.
    **kwargs :
        Keyword arguments passed to the ProbabilityDistribution constructor.
    elements : ru.UniverseAM or iterable of rankings, optional
        The set of rankings on which the distribution is concentrated. If None,
        they are sampled from the uniform distribution at the first call of `sample`, and kept afterwards.
        The default is None.
    m : int, optional
        The number of rankings on which the distribution is concentrated.
        Required if elements is None. The default is None.
    """
    def __init__(self, *args, elements: ru.UniverseAM = None, m: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._uniform = UniformDistribution(self.support, self.na, ties=self.ties, seed=self.seed)
        if elements is not None:
            elements = [_to_ranking_bytes(element) for element in elements]
            for element in elements:
                self._check_valid_element(element)
        elif m is None:
            raise ValueError("Either the elements or m must be specified.")
        self.elements = elements
        self.m = len(self.elements) if self.elements is not None else m
        self.name = f"{self.m}Degenerate"

    def _get_elements(self) -> list:
        if self.elements is None:
            self.elements = list(self._uniform.sample(self.m))
        return self.elements

    def _sample_from_support(self, n: int, **_):
        if n % self.m != 0:
            raise ValueError("n must be divisible by m.")
        return _repeat_rankings(self._get_elements(), n // self.m)

    def _sample_from_na(self, n: int, **_):
        if n % self.m != 0:
            raise ValueError("n must be divisible by m.")
        return _repeat_rankings(self._get_elements(), n // self.m)

    _sample_from_na_noties = _sample_from_na


class SpikeDistribution(ProbabilityDistribution):
    """
    Sample rankings with probability proportional to their kernel to a given center.

    This class represents a distribution where the probability of sampling a
    ranking is proportional to its kernel to a given center ranking.
    The returned sample always contains the center ranking.

    Parameters
    ----------
    *args :
        Arguments passed to the ProbabilityDistribution constructor.
    **kwargs :
        Keyword arguments passed to the ProbabilityDistribution constructor.
    center : bytes, ru.AdjacencyMatrix, or rank vector, optional
        The center ranking of the distribution. If None, it is sampled from
        the uniform distribution at the first call of `sample`, and kept afterwards. The default is None.
    kernel : ku.RankingKernel
        The kernel used to weight the rankings.
    uniform_size_sample : Union[Literal["auto", "n"], int], optional
        The size of the uniform sample used to calculate the kernels to the
        center. If "auto", the size is set to the factorial of the number of
        alternatives. If "n", the size is set to the size of the Spike sample
        as input in self._sample_from_na. If an integer, it is used as the
        sample size. The default is "n".
    """

    def __init__(self, *args, center: ru.AdjacencyMatrix = None, kernel: ku.RankingKernel,
                 uniform_size_sample: Union[Literal["auto", "n"], int] = "n", **kwargs):
        super().__init__(*args, **kwargs)
        center = _to_ranking_bytes(center)
        self._check_valid_element(center)
        self._uniform = UniformDistribution(self.support, self.na, ties=self.ties, seed=self.seed)
        self.center = center

        # size of the uniform sample used to calculate the kernels to the center
        if uniform_size_sample == "auto":
            self.uniform_size_sample = math.factorial(self.na)
        elif uniform_size_sample == "n":  # n is the size of the Spike sample as input in self._sample_from_na
            self.uniform_size_sample = "n"
        elif isinstance(uniform_size_sample, Integral) and not isinstance(uniform_size_sample, bool):
            self.uniform_size_sample = int(uniform_size_sample)
        else:
            raise ValueError(f"Invalid uniform_size_sample={uniform_size_sample}. Accepted: 'auto', 'n', or int.")
        self.kernel = kernel
        self.name = f"Spike"

    def _ntmp(self, n: int):
        """
        Size of the uniform sample used to calculate the kernels to the center.

        Parameters
        ----------
        n : int
            The size of the Spike sample.

        Returns
        -------
        int
            The size of the uniform sample.
        """
        if self.uniform_size_sample == "n":
            return n
        return self.uniform_size_sample

    def _sample_from_na(self, n: int, **_):
        """
        Samples rankings from the Spike distribution based on the number of alternatives.

        This method samples 'n' rankings from the distribution, ensuring that
        the center ranking is included in the sample. It first samples a
        uniform sample of rankings and then weights the probability of each
        ranking based on its kernel to the center.

        Parameters
        ----------
        n : int
            The number of rankings to sample.

        Returns
        -------
        ru.SampleAM
            An array of rankings sampled from the distribution.
        """
        if self.center is None:
            self.center = self._uniform.sample(1)[0]
        self.centertmp = self.center
        unif_sample = self._uniform.sample(self._ntmp(n)).append(self.centertmp)  # add center to sample
        pmf = np.array([self.kernel(self.centertmp, x, use_rv=False) for x in unif_sample])
        self.unif_sample = unif_sample
        self.pmftmp = pmf

        return ru.SampleAM(self.rng.choice(unif_sample, size=n-1, replace=True, p=pmf/pmf.sum())).append(self.centertmp)

    def _sample_from_na_noties(self, n: int, **kwargs):
        """
        Samples rankings from the Spike distribution without ties (the uniform sample has no ties).
        """
        return self._sample_from_na(n, **kwargs)


class PMFDistribution(ProbabilityDistribution):
    """
    Probability distribution defined by a custom probability mass function (PMF).

    This class represents a discrete probability distribution over a specified support,
    where each element has an explicitly defined probability mass. A support must be
    provided, and its length must match the length of the PMF.

    Parameters
    ----------
    pmf : np.ndarray
        Array of probability masses corresponding to elements in the support.
    support : ru.SampleAM
        The rankings with positive probability.
    *args : tuple
        Additional positional arguments passed to the parent class.
    **kwargs : dict
        Additional keyword arguments passed to the parent class (e.g., seed).

    Raises
    ------
    ValueError
        If the length of the support and the PMF do not match.

    Examples
    --------
    >>> distr = PMFDistribution.from_sample(sample, seed=0)   # empirical distribution of `sample`
    >>> resample = distr.sample(50)                            # 50 rankings drawn with replacement
    """

    def __init__(self, pmf: np.ndarray, support: ru.SampleAM, *args, **kwargs):
        super().__init__(support, *args, **kwargs)
        self.pmf = np.asarray(pmf, dtype=float)
        self.name = "PMF"
        self.support = support

        if len(self.support) != len(self.pmf):
            raise ValueError("The length of support and pmf must coincide.")

    @classmethod
    def from_sample(cls, sample: ru.SampleAM, **kwargs):
        """Empirical distribution of `sample`. kwargs (e.g., seed) are passed to the constructor."""
        support, pmf = sample.get_support_pmf()
        return cls(support=support, pmf=pmf, **kwargs)

    def _sample_from_na(self, n: int, **kwargs):
        raise NotImplementedError("Not possible to sample without a support.")

    def sample(self, n: int, **kwargs) -> ru.SampleAM:
        """Sample n rankings, with replacement, according to the pmf."""
        return ru.SampleAM(self.rng.choice(self.support, n, replace=True, p=self.pmf/self.pmf.sum()))

    def __str__(self):
        return f"PMF(na={self.na}, ties={self.ties}, pmf={self.pmf})"



###################################################
# The following distributions are not up to date
###################################################


# class BallDistribution(ProbabilityDistribution):
#
#     def __init__(self, *args, center: ru.AdjacencyMatrix = None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._check_valid_element(center)
#         raise NotImplementedError
#
#
# class BallProbabilityDistribution(ProbabilityDistribution):
#     """
#     Samples uniformly from points with kernel from center greater/smaller than radius.
#     """
#
#     def __init__(self, support: ru.UniverseAM, dimension: int):
#         super().__init__(support, dimension)
#         raise NotImplementedError
#
#     def sample(self, n: int, seed: int = 42, center=None, radius: float = 0,
#                kind: Literal["ball", "antiball"] = "ball",
#                kernel: ku.Kernel = lambda x, y: np.all(x == y).astype(int),
#                **kernelargs) -> ru.SampleAM:
#
#         # if you know what center you want, use that one
#         if center is not None:
#             if center not in self.support:
#                 raise ValueError("If center is not None, it must belong to self.support.")
#         # otherwise, use a random one
#         else:
#             print("center?")
#             center = np.random.default_rng(seed).choice(self.support, size=1)[0]
#
#         if kind == "ball":
#             c = np.greater_equal
#         elif kind == "antiball":
#             c = np.less_equal
#         else:
#             raise ValueError("Invalid value for parameter kind.")
#
#         self.distr = FunctionDefaultDict(lambda x: 1 if c(kernel(center, x, **kernelargs), radius) else 0)
#         small_support = np.array([x for x in self.support
#                                    if c(kernel(center, x, **kernelargs), radius)], dtype=object)
#
#         return ru.SampleAM(np.random.default_rng(seed).choice(small_support, size=n, replace=True))
#
#     def lazy_sample(self, n: int, max_steps = 1, seed: int = 42, center=None, radius: float = 0,
#                     kind: Literal["ball", "antiball"] = "ball",
#                     kernel: ku.Kernel = ku.trivial_kernel,
#                     **kernelargs) -> ru.SampleAM:
#
#         rng = np.random.default_rng(seed)
#
#         # if you know what center you want, use that one
#         if center is not None:
#             assert len(center) == self.na
#             # convert to valid rv
#             center = rlu.vec2rv(center)
#         # otherwise, use a random one
#         else:
#             center = rlu.vec2rv(rng.integers(low=0, high=self.na, size=self.na))
#
#         if kind == "ball":
#             c = np.greater_equal
#         elif kind == "antiball":
#             c = np.less_equal
#         else:
#             raise ValueError("Invalid value for parameter kind.")
#
#         samples = []
#         ctr = 0
#         while len(samples) < n:
#             if ctr >= max_steps:
#                 break
#             if max_steps is not None:
#                 ctr += 1
#             random_vector = rng.integers(low=0, high=self.na, size=self.na)
#             valid_rv = rlu.vec2rv(random_vector)
#             condition = c(kernel(center, valid_rv, **kernelargs), radius)
#
#             if condition:
#                 samples.append(random_vector)
#
#         samples = np.column_stack(samples) if samples else np.empty((self.na, 0))
#
#         out = ru.SampleAM.from_rank_vector_matrix(samples)
#         out.rv = samples
#
#         return out