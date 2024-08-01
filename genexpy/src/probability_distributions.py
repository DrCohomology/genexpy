import warnings

import numpy as np
import time

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from scipy.special import factorial, stirling2
from tqdm import tqdm
from typing import Literal

from . import kernels as ku
from . import rankings_utils as ru
from . import relation_utils as rlu


def sample_from_sphere(na: int, n: int, rng: np.random.Generator) -> np.ndarray[float]:
    """
    sample points uniformly from the unitary na-sphere.
    Credits to https://mathoverflow.net/questions/24688/efficiently-sampling-points-uniformly-from-the-surface-of-an-n-sphere.
    na: sphere dimensionality
    n: sample size
    rng: random number generator
    returns a "list" of points uniformly from the unitary na-sphere.
    """
    x = rng.normal(0, 1, (na, n))
    return x / np.linalg.norm(x, axis=1).reshape(-1, 1)


def get_unique_ranks_distribution(n, exact=False):
    """
    A ranking with ties has a different number of unique ranks. For instance, 0112 has 3 unique ranks.
    Given a ranking of 'n' alternatives, get the number of rankings with k unique ranks for all 1 <= k <= n.
        closely related to T(n, k) in https://oeis.org/A019538
    return the normalized value.
    """
    out = factorial(np.arange(n)+1, exact=exact) * stirling2(n, np.arange(n)+1, exact=exact)
    return out / out.sum()


class FunctionDefaultDict(defaultdict):
    def __init__(self, func, *args, **kwargs):
        super().__init__(func, *args, **kwargs)
        self.func = func

    def __missing__(self, key):
        return self.func(key)


class ProbabilityDistribution(ABC):
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
        if x is not None:
            if self.universe is not None and x not in self.universe:
                raise ValueError("The input element must belong to the universe.")
            else:
                if len(x) != self.na ** 2:  # bytestring representation:
                    raise ValueError("The input element must have the correct number of alternatives.")
                # TODO: check if ties are present

    def _sample_from_universe(self, n: int, **kwargs):
        return self.universe.get_subsample(subsample_size=n, seed=self.seed, use_key=False, replace=True)

    @abstractmethod
    def _sample_from_na(self, n: int, **kwargs):
        pass

    def _sample_from_na_noties(self, n: int, **kwargs):
        raise NotImplementedError

    def sample(self, n: int, **kwargs) -> ru.SampleAM:
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
        return f"{self.name}(na={self.na}, ties={self.ties})"

class UniformDistribution(ProbabilityDistribution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.pmf = FunctionDefaultDict(lambda x: 1 / self.na)
        self.name = "Uniform"

    def _sample_from_na(self, n: int, **kwargs) -> ru.SampleAM:
        """
        Sample uniformly from rankings of a given number of alternatives.
        how to:
            1. sample the number of tied pairs with probability proportional to the corresponding number of rankings
            2. sample uniformly points from the unitary sphere
            3. convert the sample on a sphere to a sample on the rankings
        updated:
            1. sample the number of unique ranks with probability proportional to the corresponding number of rankings
            2. shuffle the unique ranks
            3. assign to each unique rank a list of indices in the output ranking
        """
        # self.sphere_sample = sample_from_sphere(n, self.na, self.rng)
        # self.rankings = rlu.vecs2rv(self.sphere_sample, lower_is_better=True)
        # return ru.SampleAM.from_rank_function_matrix(self.rankings.T)

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
        return ru.SampleAM.from_rank_function_matrix(np.array(rf).T)

    def _sample_from_na_noties(self, n: int, **kwargs) -> ru.SampleAM :
        """
        Generate permutations of a range(0, na)
        """
        return ru.SampleAM.from_rank_function_matrix(
            self.rng.permuted(np.tile(np.arange(self.na), n).reshape(n, self.na), axis=1).T)


class DegenerateDistribution(ProbabilityDistribution):
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

    def __init__(self, *args, center: ru.AdjacencyMatrix = None, kernel: ku.Kernel = ku.mallows_kernel,
                 kernelargs: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_valid_element(center)
        self._uniform = UniformDistribution(self.universe, self.na, ties=self.ties, seed=self.seed)
        self.center = center if center is not None else self._uniform.sample(1)[0]
        self.kernel = kernel
        self.kernelargs = kernelargs or {}
        self.name = f"Spike"

    def _sample_from_na(self, n: int, **_):
        """
        sample uniformly, then sample from the sample weighting with the kernel
        """
        unif_sample = self._uniform.sample(n)
        pmf = np.array([self.kernel(self.center, x, **self.kernelargs) for x in unif_sample])
        self.unif_sample = unif_sample
        self.pmf = pmf

        return ru.SampleAM(self.rng.choice(unif_sample, size=n, replace=True, p=pmf/pmf.sum()))

    def _sample_from_na_noties(self, n: int, **kwargs):
        """
        sample uniformly, then sample from the sample weighting with the kernel
        """
        unif_sample = self._uniform.sample(n)
        pmf = np.array([self.kernel(self.center, x, **self.kernelargs) for x in unif_sample])
        self.unif_sample = unif_sample
        self.pmf = pmf

        return ru.SampleAM(self.rng.choice(unif_sample, size=n, replace=True, p=pmf/pmf.sum()))


class PMFDistribution(ProbabilityDistribution):
    """
    With custom probability mass function.
    Requires a universe.
    """

    def __init__(self, pmf: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pmf = pmf
        self.name = "PMF"

        if self.universe is None:
            raise ValueError("Universe must be specified for a PMFDistribution.")
        if len(self.universe) != len(self.pmf):
            raise ValueError("The length of universe and pmf must coincide.")

    def _sample_from_na(self, n: int, **kwargs):
        raise NotImplementedError("Not possible to sample without a universe.")

    def sample(self, n: int, **kwargs) -> ru.SampleAM:
        return ru.SampleAM(np.random.default_rng(self.seed).choice(self.universe, n, replace=True, p=self.pmf/self.pmf.sum()))







#
#
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
#         out = ru.SampleAM.from_rank_function_matrix(samples)
#         out.rv = samples
#
#         return out