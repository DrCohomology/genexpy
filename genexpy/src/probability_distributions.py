"""
Utility module with probability distributions over rankings.
"""
import builtins
import math
import numpy as np
import time

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from scipy.special import factorial, stirling2
from typing import Literal, Union

from . import kernels as ku
from . import rankings_utils as ru


def get_unique_ranks_distribution(n, exact=False, normalized=True):
    """
    Calculates the distribution of unique ranks in rankings of length 'n'.

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
    The distribution is calculated using the following formula:

    .. math::
        P(k) = \frac{n! S(n, k)}{n!}

    where :math:`S(n, k)` is the Stirling number of the second kind, representing
    the number of ways to partition a set of 'n' elements into 'k' non-empty
    subsets.

    The terms n(n-1)/2 + 1 to n(n+1)/2 in T(n, k) in https://oeis.org/A019538
    correspond to the number of rankings with k unique ranks.
    """
    out = factorial(np.arange(n)+1, exact=exact) * stirling2(n, np.arange(n)+1, exact=exact)
    out = out.astype(float)
    return out / out.sum() if normalized else out


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
    universe : ru.SampleAM, optional
        The universe of possible rankings. If None, the universe is defined by
        the number of alternatives (na). The default is None.
    na : int, optional
        The number of alternatives in the ranking. Required if universe is None.
        The default is None.
    ties : bool, optional
        Whether ties are allowed in the rankings. The default is True.
    seed : int, optional
        The random seed for sampling. The default is None.

    Attributes
    ----------
    universe : ru.SampleAM
        The universe of possible rankings.
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
    """

    def __init__(self, universe: ru.SampleAM = None, na: int = None, ties: bool = True, seed: int = None):
        self.universe = universe
        if universe is None:
            if na is None:
                raise ValueError("Specify the number of alternatives or a universe")
            self.na = na
        else:
            self.na = universe.get_na()
            if len(self.universe) == 0:
                raise ValueError("The input list is empty.")

        self.pmf = defaultdict(lambda: 0)  # probability mass function, used by some distributions
        self.ties = ties
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sample_time = np.nan
        self.name = "Generic"

    def _check_valid_element(self, x):
        """
        Checks if an element is valid for the distribution.

        Parameters
        ----------
        x : ru.AdjacencyMatrix or bytes
            The element to check.

        Raises
        ------
        ValueError
            If the element is not valid for the distribution.
        """
        if x is not None:
            if self.universe is not None and x not in self.universe:
                raise ValueError("The input element must belong to the universe.")
            else:
                if len(x) != self.na ** 2:  # bytestring representation:
                    raise ValueError("The input element must have the correct number of alternatives.")
                # TODO: check if ties are present

    def _sample_from_universe(self, n: int, **kwargs):
        """
        Samples rankings from the universe.

        Parameters
        ----------
        n : int
            The number of rankings to sample.

        Returns
        -------
        ru.SampleAM
            A sample of rankings from the universe.
        """
        return self.universe.get_subsample(subsample_size=n, seed=self.seed, use_key=False, replace=True)

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
        if self.universe is not None:
            out = self._sample_from_universe(n, **kwargs)
        else:
            if self.ties:
                out = self._sample_from_na(n, **kwargs)
            else:
                out = self._sample_from_na_noties(n, **kwargs)
        self.sample_time = time.time() - start_time
        return out

    def __str__(self):
        """Returns a string representation of the distribution."""
        return f"{self.name}(na={self.na}, ties={self.ties})"


class UniformDistribution(ProbabilityDistribution):
    """
    Uniform distribution over rankings.

    This class represents a uniform distribution over all possible rankings
    of a given number of alternatives.

    Parameters
    ----------
    *args :
        Arguments passed to the ProbabilityDistribution constructor.
    **kwargs :
        Keyword arguments passed to the ProbabilityDistribution constructor.

    Methods
    -------
    _sample_from_na(n: int, **kwargs) -> ru.SampleAM
        Samples rankings from the uniform distribution based on the number of alternatives.
    _sample_from_na_noties(n: int, **kwargs) -> ru.SampleAM
        Samples rankings from the uniform distribution without ties.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.pmf = FunctionDefaultDict(lambda x: 1 / self.na)
        self.name = "Uniform"

    def _sample_from_na(self, n: int, **kwargs) -> ru.SampleAM:
        """
        Samples rankings from the uniform distribution based on the number of alternatives.

        This method uses a sampling strategy based on the number of unique ranks
        in the rankings.

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
        rf = []
        for nur in nurs:
            # create an array of length n. Then, for all ranks, sample indices from a pool and assign that rank
            pool = np.arange(self.na)
            out = np.empty(self.na, dtype=int)
            for ir, rank in enumerate(self.rng.choice(np.arange(nur), replace=False, size=nur)):  # shuffle the ranks
                # last iteration: assign rank to remaning indices
                if ir == nur - 1:
                    out[pool] = rank
                    break

                idx = self.rng.choice(pool, replace=True, size=len(pool) - (nur - ir) + 1)
                out[idx] = rank
                pool = np.setdiff1d(pool, idx)

            assert np.isin(np.arange(nur), out).all(), "Not all ranks were used"

            rf.append(out)
        return ru.SampleAM.from_rank_vector_matrix(np.array(rf).T)

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
    element : ru.AdjacencyMatrix, optional
        The ranking on which the distribution is concentrated. If None, it is
        sampled from the uniform distribution. The default is None.

    Methods
    -------
    _sample_from_na(n: int, **kwargs) -> ru.SampleAM
        Samples rankings from the degenerate distribution based on the number of alternatives.
    _sample_from_na_noties(n: int, **kwargs) -> ru.SampleAM
        Samples rankings from the degenerate distribution without ties.
    """
    def __init__(self, *args, element: ru.AdjacencyMatrix = None, **kwargs):
        super().__init__(*args, **kwargs)
        # self.pmf = FunctionDefaultDict(lambda x: 1 / self.na)
        self._check_valid_element(element)
        self._uniform = UniformDistribution(self.universe, self.na, ties=self.ties, seed=self.seed)
        self.element = element
        self.name = "Degenerate"

    def _sample_from_universe(self, n: int, **kwargs):
        if self.element is not None:
            return ru.SampleAM(np.array([self.element]*n))
        else:
            return np.tile(UniformDistribution(universe=self.universe, na=self.na, seed=self.seed).sample(1), n)

    def _sample_from_na(self, n: int, **kwargs):
        if self.element is not None:
            return ru.SampleAM(np.array([self.element] * n))
        else:
            return np.tile(self._uniform.sample(1), n)


class MDegenerateDistribution(ProbabilityDistribution):
    """
    Multi-degenerate distribution concentrated on multiple rankings.

    This class represents a distribution where all probability mass is
    concentrated on a set of 'm' rankings.

    Parameters
    ----------
    *args :
        Arguments passed to the ProbabilityDistribution constructor.
    **kwargs :
        Keyword arguments passed to the ProbabilityDistribution constructor.
    elements : ru.UniverseAM, optional
        The set of rankings on which the distribution is concentrated. If None,
        it is sampled from the uniform distribution. The default is None.
    m : int, optional
        The number of rankings on which the distribution is concentrated.
        Required if elements is None. The default is None.

    Methods
    -------
    _sample_from_na(n: int, **kwargs) -> ru.SampleAM
        Samples rankings from the multi-degenerate distribution based on the number of alternatives.
    _sample_from_na_noties(n: int, **kwargs) -> ru.SampleAM
        Samples rankings from the multi-degenerate distribution without ties.
    """
    def __init__(self, *args, elements: ru.UniverseAM = None, m: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        # self.pmf = FunctionDefaultDict(lambda x: 1 / self.na)
        self._uniform = UniformDistribution(self.universe, self.na, ties=self.ties, seed=self.seed)
        self.elements = elements
        if self.elements is not None:
            for element in self.elements:
                self._check_valid_element(element)
        else:
            if m is None:
                raise ValueError("Either the elements or m must be specified.")
        self.m = len(self.elements) if self.elements else m
        self.name = f"{self.m}Degenerate"

    def _sample_from_universe(self, n: int, **_):
        assert n % self.m == 0, "n must be divisible by m."
        if self.elements is not None:
            return np.tile(self.elements, n // self.m)
        else:
            elements = UniformDistribution(universe=self.universe, na=self.na, seed=self.seed).sample(self.m)
            return np.tile(elements, n // self.m)

    def _sample_from_na(self, n: int, **_):
        assert n % self.m == 0, "n must be divisible by m."
        if self.elements is not None:
            return np.tile(self.elements, n // self.m)
        else:
            elements = self._uniform.sample(self.m)
            return np.tile(elements, n // self.m)



class SpikeDistribution(ProbabilityDistribution):
    """
    Sample rankings with probability proportional to their kernel to a given center.

    This class represents a distribution where the probability of sampling a
    ranking is proportional to its kernel distance to a given center ranking.
    The returned sample always contains the center ranking.

    Parameters
    ----------
    *args :
        Arguments passed to the ProbabilityDistribution constructor.
    **kwargs :
        Keyword arguments passed to the ProbabilityDistribution constructor.
    center : ru.AdjacencyMatrix, optional
        The center ranking of the distribution. If None, it is sampled from
        the uniform distribution. The default is None.
    kernel : ku.Kernel, optional
        The kernel function used to calculate distances to the center. The
        default is ku.mallows_kernel.
    kernelargs : dict, optional
        Additional arguments to pass to the kernel function. The default is
        None.
    uniform_size_sample : Union[Literal["auto", "n"], int], optional
        The size of the uniform sample used to calculate the kernels to the
        center. If "auto", the size is set to the factorial of the number of
        alternatives. If "n", the size is set to the size of the Spike sample
        as input in self._sample_from_na. If an integer, it is used as the
        sample size. The default is "n".

    Methods
    -------
    _sample_from_na(n: int, **kwargs) -> ru.SampleAM
        Samples rankings from the Spike distribution based on the number of alternatives.
    _sample_from_na_noties(n: int, **kwargs) -> ru.SampleAM
        Samples rankings from the Spike distribution without ties.
    """

    def __init__(self, *args, center: ru.AdjacencyMatrix = None, kernel: ku.Kernel = ku.mallows_kernel,
                 kernelargs: dict = None, uniform_size_sample: Union[Literal["auto", "n"], int] = "n", **kwargs):
        super().__init__(*args, **kwargs)
        self._check_valid_element(center)
        self._uniform = UniformDistribution(self.universe, self.na, ties=self.ties, seed=self.seed)
        self.center = center

        match uniform_size_sample:  # size of the uniform sample used to calculate the kernels to the center
            case "auto": self.uniform_size_sample = math.factorial(self.na)
            case "n": self.uniform_size_sample = "n"  # n is the size of the Spike sample as input in self._sample_from_na
            case builtins.int: self.uniform_size_sample = uniform_size_sample
        self.uniform_size_sample = math.factorial(self.na) if uniform_size_sample == "auto" else uniform_size_sample
        self.kernel = kernel
        # the kernel must have use_rv=False
        if kernelargs is None:
            self.kernelargs = {"use_rv": False}
        else:
            self.kernelargs = kernelargs.update({"use_rv": False})
        self.name = f"Spike"


    def _ntmp(self, n: int):
        """
        Helper function to determine the uniform sample size.

        This function determines the size of the uniform sample used to
        calculate the kernels to the center based on the value of
        `uniform_size_sample`.

        Parameters
        ----------
        n : int
            The size of the Spike sample.

        Returns
        -------
        int
            The size of the uniform sample.
        """
        # get the sample size from the uniform distribution
        if self.uniform_size_sample == "n":
            return n
        elif isinstance(self.uniform_size_sample, int):
            return self.uniform_size_sample
        else:
            raise ValueError(f"Unsupported uniform_size_sample with type {type(self.uniform_size_sample)}")


    def _sample_from_na(self, n: int, **_):
        """
        Samples rankings from the Spike distribution based on the number of alternatives.

        This method samples 'n' rankings from the distribution, ensuring that
        the center ranking is included in the sample. It first samples a
        uniform sample of rankings and then weights the probability of each
        ranking based on its kernel distance to the center.

        Parameters
        ----------
        n : int
            The number of rankings to sample.

        Returns
        -------
        ru.SampleAM
            An array of rankings sampled from the distribution.
        """
        self.centertmp = self.center or self._uniform.sample(1)[0]
        unif_sample = self._uniform.sample(self._ntmp(n)).append(self.centertmp)  # add center to sample
        pmf = np.array([self.kernel(self.centertmp, x, **self.kernelargs) for x in unif_sample])
        self.unif_sample = unif_sample
        self.pmftmp = pmf

        return ru.SampleAM(self.rng.choice(unif_sample, size=n-1, replace=True, p=pmf/pmf.sum())).append(self.centertmp)

    def _sample_from_na_noties(self, n: int, **kwargs):
        """
        Samples rankings from the Spike distribution without ties.

        This method samples 'n' rankings from the distribution without ties,
        ensuring that the center ranking is included in the sample. It first
        samples a uniform sample of rankings without ties and then weights the
        probability of each ranking based on its kernel distance to the
        center.

        Parameters
        ----------
        n : int
            The number of rankings to sample.

        Returns
        -------
        ru.SampleAM
            An array of rankings sampled from the distribution without ties.
        """
        # unif_sample = self._uniform.sample(self._ntmp(n)).merge(self.center)  # add center to sample
        # pmf = np.array([self.kernel(self.center, x, **self.kernelargs) for x in unif_sample])
        # self.unif_sample = unif_sample
        # self.pmf = pmf
        #
        # return ru.SampleAM(self.rng.choice(unif_sample, size=n, replace=True, p=pmf/pmf.sum()))
        return self._sample_from_na(n, **kwargs)


class PMFDistribution(ProbabilityDistribution):
    """
    Probability distribution defined by a custom probability mass function (PMF).

    This class represents a discrete probability distribution over a specified universe,
    where each element has an explicitly defined probability mass. A universe must be
    provided, and its length must match the length of the PMF.

    Parameters
    ----------
    pmf : np.ndarray
        Array of probability masses corresponding to elements in the universe.
    *args : tuple
        Additional positional arguments passed to the parent class.
    **kwargs : dict
        Additional keyword arguments passed to the parent class. Must include 'universe'.

    Raises
    ------
    ValueError
        If the universe is not specified.
    ValueError
        If the length of the universe and the PMF do not match.

    Attributes
    ----------
    pmf : np.ndarray
        The probability mass function array.
    name : str
        Name of the distribution, set to "PMF".

    Methods
    -------
    from_sample(sample, **kwargs)
        Creates a PMFDistribution from a sample by extracting its PMF and universe.
    sample(n, **kwargs)
        Generates a sample of size `n` based on the PMF.
    __str__()
        Returns a string representation of the distribution.
    """


    def __init__(self, pmf: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pmf = pmf
        self.name = "PMF"

        if self.universe is None:
            raise ValueError("Universe must be specified for a PMFDistribution.")
        if len(self.universe) != len(self.pmf):
            raise ValueError("The length of universe and pmf must coincide.")

    @classmethod
    def from_sample(cls, sample: ru.SampleAM, **kwargs):
        universe, pmf = sample.get_universe_pmf()
        return PMFDistribution(universe=universe, pmf=pmf, **kwargs)

    def _sample_from_na(self, n: int, **kwargs):
        raise NotImplementedError("Not possible to sample without a universe.")

    def sample(self, n: int, **kwargs) -> ru.SampleAM:
        return ru.SampleAM(self.rng.choice(self.universe, n, replace=True, p=self.pmf/self.pmf.sum()))

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
#     def __init__(self, universe: ru.UniverseAM, dimension: int):
#         super().__init__(universe, dimension)
#         raise NotImplementedError
#
#     def sample(self, n: int, seed: int = 42, center=None, radius: float = 0,
#                kind: Literal["ball", "antiball"] = "ball",
#                kernel: ku.Kernel = lambda x, y: np.all(x == y).astype(int),
#                **kernelargs) -> ru.SampleAM:
#
#         # if you know what center you want, use that one
#         if center is not None:
#             if center not in self.universe:
#                 raise ValueError("If center is not None, it must belong to self.universe.")
#         # otherwise, use a random one
#         else:
#             print("center?")
#             center = np.random.default_rng(seed).choice(self.universe, size=1)[0]
#
#         if kind == "ball":
#             c = np.greater_equal
#         elif kind == "antiball":
#             c = np.less_equal
#         else:
#             raise ValueError("Invalid value for parameter kind.")
#
#         self.distr = FunctionDefaultDict(lambda x: 1 if c(kernel(center, x, **kernelargs), radius) else 0)
#         small_universe = np.array([x for x in self.universe
#                                    if c(kernel(center, x, **kernelargs), radius)], dtype=object)
#
#         return ru.SampleAM(np.random.default_rng(seed).choice(small_universe, size=n, replace=True))
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