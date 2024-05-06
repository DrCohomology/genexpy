import numpy as np

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Literal

from . import kernels as ku
from . import rankings_utils as ru
from . import relation_utils as rlu


class FunctionDefaultDict(defaultdict):
    def __init__(self, func, *args, **kwargs):
        super().__init__(func, *args, **kwargs)
        self.func = func

    def __missing__(self, key):
        return self.func(key)


class ProbabilityDistribution(ABC):
    def __init__(self, universe: ru.UniverseAM = None, na: int = None):
        if universe is None:
            if na is None:
                raise ValueError("Specify the number of alternatives or a universe")
            self.na = na
        else:
            self.na = universe.get_na()
        self.universe = universe
        if self.universe is not None and len(self.universe) == 0:
            raise ValueError("The input list is empty.")
        self.distr = defaultdict(lambda: 0)

    @abstractmethod
    def sample(self, n: int, seed: int = 42) -> np.array:
        pass


class UniformDistribution(ProbabilityDistribution):
    def __init__(self, universe: ru.UniverseAM):
        super().__init__(universe)

    def sample(self, n: int, seed: int = 42) -> ru.SampleAM:
        """
        Sample uniformly from a list of adjacency matrices multiple times with replacement.

        Args:
            n (int): Number of samples to generate.

        Returns:
            list: A list of n random adjacency matrices sampled from the list.
        """
        self.distr = defaultdict(lambda x: 1 / len(self.universe))
        return ru.SampleAM(np.random.default_rng(seed).choice(self.universe, size=n, replace=True))


class DegenerateDistribution(ProbabilityDistribution):
    def __init__(self, universe: ru.UniverseAM):
        super().__init__(universe)

    def sample(self, n: int, seed: int = 42, element=None) -> ru.SampleAM:
        """
        Sample a single ranking uniformly at random from a list of rankings (represented as
        adjacency matrices) and replicate it 'n' times. This is a degenerate sampling strategy.

        Args:
            n (int): Number of samples to generate. Each sample will be a copy of the same
                     randomly chosen adjacency matrix.
            seed (int, optional): Seed for the random number generator. Default is 42.
            elemeent (): object in the universe

        Returns:
            np.array: A list containing 'n' copies of the same randomly chosen
                              adjacency matrix from the input list.

        Raises:
            ValueError: If the input list is empty.
        """

        # if you know what element you want, return that one
        if element is not None:
            if element not in self.universe:
                raise ValueError("If element is not None, it must belong to self.universe.")
        # otherwise, return a random one
        else:
            element = np.random.default_rng(seed).choice(self.universe, size=1)[0]

        self.distr.update({element: 1})
        return ru.SampleAM(np.array([element] * n, dtype=self.universe.dtype))


class MDegenerateDistribution(ProbabilityDistribution):
    def __init__(self, universe: ru.UniverseAM):
        super().__init__(universe)

    def sample(self, n: int, seed: int = 42, m: int = 2, elements: ru.UniverseAM = tuple()) -> ru.SampleAM:
        """
        Sample bidegenerately from a list of NumPy arrays. The first half of the samples will be
        the array at the 'first' index of the list, and the second half will be the array at the
        'second' index. The resulting list is then shuffled.

        Args:
            n (int): Total number of samples to generate. Must be a positive even number.
            seed (int): Seed for the random number generator. Default is 42.
            first (int): Index of the first array to sample in the list. Default is 0.
            second (int): Index of the second array to sample in the list. Default is 1.

        Returns:
            List[np.ndarray]: A list of 'n' NumPy arrays sampled bidegenerately from the list.

        Raises:
            ValueError: If the input list is empty, 'n' is not a positive even number, or
                        if the list has less than two distinct elements.
                        :param m: number of elements we are sampling uniformly
                        :type m: int
                        :param elements: known elements
                        :type elements:
        """
        if n <= 0 or n % m != 0:
            raise ValueError("n must be a positive number with n % m == 0.")
        if not set(elements).issubset(self.universe):
            raise ValueError("The elements must a subset of self.universe.")
        if len(elements) > m:
            raise ValueError("You specificed more elements than can be output.")

        small_universe = list(elements)
        small_universe.extend(np.random.default_rng(seed).choice(self.universe, m - len(elements)).astype(object))
        small_universe = np.array(small_universe, dtype=object)

        self.distr.update({x: 1 / m for x in small_universe})

        return ru.SampleAM(np.random.default_rng(seed).permutation(np.repeat(small_universe, n // m)))


class CenterProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, universe: ru.UniverseAM):
        super().__init__(universe)

    def sample(self, n: int, seed: int = 42, center=None,
               kernel: ku.Kernel = ku.trivial_kernel,
               **kernelargs) -> ru.SampleAM:

        # if you know what center you want, use that one
        if center is not None:
            if center not in self.universe:
                raise ValueError("If center is not None, it must belong to self.universe.")
        # otherwise, use a random one
        else:
            center = np.random.default_rng(seed).choice(self.universe, size=1)[0]

        self.distr = FunctionDefaultDict(lambda x: kernel(center, x, **kernelargs))
        probs = np.array([kernel(center, x, **kernelargs) for x in self.universe])
        probs /= probs.sum()

        return ru.SampleAM(
            np.random.default_rng(seed).choice(self.universe, size=n, replace=True, p=probs).astype(object))


class BallProbabilityDistribution(ProbabilityDistribution):
    """
    Samples uniformly from points with kernel from center greater/smaller than radius.
    """

    def __init__(self, universe: ru.UniverseAM, dimension: int):
        super().__init__(universe, dimension)

    def sample(self, n: int, seed: int = 42, center=None, radius: float = 0,
               kind: Literal["ball", "antiball"] = "ball",
               kernel: ku.Kernel = lambda x, y: np.all(x == y).astype(int),
               **kernelargs) -> ru.SampleAM:

        # if you know what center you want, use that one
        if center is not None:
            if center not in self.universe:
                raise ValueError("If center is not None, it must belong to self.universe.")
        # otherwise, use a random one
        else:
            print("center?")
            center = np.random.default_rng(seed).choice(self.universe, size=1)[0]

        if kind == "ball":
            c = np.greater_equal
        elif kind == "antiball":
            c = np.less_equal
        else:
            raise ValueError("Invalid value for parameter kind.")

        self.distr = FunctionDefaultDict(lambda x: 1 if c(kernel(center, x, **kernelargs), radius) else 0)
        small_universe = np.array([x for x in self.universe
                                   if c(kernel(center, x, **kernelargs), radius)], dtype=object)

        return ru.SampleAM(np.random.default_rng(seed).choice(small_universe, size=n, replace=True))

    def lazy_sample(self, n: int, max_steps = 1, seed: int = 42, center=None, radius: float = 0,
                    kind: Literal["ball", "antiball"] = "ball",
                    kernel: ku.Kernel = ku.trivial_kernel,
                    **kernelargs) -> ru.SampleAM:

        rng = np.random.default_rng(seed)

        # if you know what center you want, use that one
        if center is not None:
            assert len(center) == self.na
            # convert to valid rv
            center = rlu.vec2rv(center)
        # otherwise, use a random one
        else:
            center = rlu.vec2rv(rng.integers(low=0, high=self.na, size=self.na))

        if kind == "ball":
            c = np.greater_equal
        elif kind == "antiball":
            c = np.less_equal
        else:
            raise ValueError("Invalid value for parameter kind.")

        samples = []
        ctr = 0
        while len(samples) < n:
            if ctr >= max_steps:
                break
            if max_steps is not None:
                ctr += 1
            random_vector = rng.integers(low=0, high=self.na, size=self.na)
            valid_rv = rlu.vec2rv(random_vector)
            condition = c(kernel(center, valid_rv, **kernelargs), radius)

            if condition:
                samples.append(random_vector)
        
        samples = np.column_stack(samples) if samples else np.empty((self.na, 0))
        
        out = ru.SampleAM.from_rank_function_matrix(samples)
        out.rv = samples

        return out