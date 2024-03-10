import math
import numpy as np
import pandas as pd

from collections.abc import Collection
from itertools import permutations
from tqdm import tqdm
from typing import AnyStr, Iterable

import src.relation_utils as rlu


class AdjacencyMatrix(np.ndarray):
    """
    Store a ranking as an adjacency matrix.
    M[i, j] = int(R[i] <= R[j])
    """

    def __new__(cls, input_array):
        assert len(input_array.shape) == 2, "Wrong number of dimensions."
        assert input_array.shape[0] == input_array.shape[1], "An adjacency matrix is always square."
        assert (input_array == input_array.astype(bool).astype(int)).all(), "Matrix is not boolean."
        return np.asarray(input_array).view(cls)

    @classmethod
    def from_rank_function(cls, rf: Iterable):
        """
        a rank function maps an alternative into its rank
        """
        return np.array([[ri <= rj for rj in rf] for ri in rf]).astype(int).view(cls)

    @classmethod
    def from_bytes(cls, bytestring: bytes, shape: Iterable[int]):
        """
        bytestring is a string of bytes, but encoded as an object.
        This is because np.tobytes() will output bytestrings without ending 0's. However, we need the ending 0's for
            np.frombuffer.
        """
        return np.frombuffer(bytestring, dtype=np.int8).reshape(shape).view(cls)

    def _list(self):
        pass

    def tohashable(self):
        return self.astype(np.int8).tobytes()

    def __hash__(self):
        return hash(self.tohashable())

    def __iter__(self):
        pass


class UniverseAM(np.ndarray):
    """
    Universe of AdjacencyMatrix objects.
    It's a np.array storing the binary encodings of adjacency matrices.
    The dtype of the array is object and not bytes because bytes removes the ending 0's, which messes up
        the reconstruction of the array using np.frombuffer. See AdjacencyMatrix.from_bytes.
    """

    def __new__(cls, input_iter: Iterable):
        try:
            return np.asarray([x.tohashable() for x in input_iter], dtype=object).view(cls)
        except AttributeError:
            if isinstance(input_iter, np.ndarray) or isinstance(input_iter, UniverseAM):
                return input_iter.view(cls)
            raise ValueError("Invalid input to UniverseAM.")

    def to_array_list(self, shape: Iterable[int]):
        # return [AdjacencyMatrix(np.frombuffer(x, dtype=int).reshape(shape)) for x in self]
        return [AdjacencyMatrix.from_bytes(x, shape) for x in self]

    def __contains__(self, bstring):
        """
        Bytestrings automatically remove ending 0's, leading to problems when trying to use np.frombuffer().
        Instead, store everything as object.
        """
        return np.any(np.isin(self, bstring))

    def _get_na_nv(self):
        """Number of alternatives = methods."""
        na = np.sqrt(len(self[0]))
        assert na == int(na), "Wrong length"
        self.na = int(na)
        self.nv = len(self)

    def get_na(self):
        self._get_na_nv()
        return self.na


class SampleAM(UniverseAM):

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def from_rank_function_dataframe(cls, rf: pd.DataFrame):
        """
        Each row of rf is an alternative.
        Each column of rf is an experimental condition/ voter.
        """
        out = np.empty_like(rf.columns)
        for ic, col in enumerate(rf.columns):
            out[ic] = AdjacencyMatrix.from_rank_function(rf[col]).tohashable()
        return out.view(cls)

    @classmethod
    def from_rank_function_matrix(cls, rf_matrix):
        """
        Convert a rank function matrix into a SampleAM object.
        Each row of rf_matrix is an alternative.
        Each column of rf_matrix is an experimental condition/ voter.
        """
        # Initialize an array to store hashable representations of adjacency matrices
        out = np.empty(rf_matrix.shape[1], dtype=object)  # Assuming rf_matrix.shape[1] is the number of columns/voters

        # Iterate through each experimental condition/voter
        for ic in range(rf_matrix.shape[1]):
            # Extract the rank function for the current column
            rank_function = rf_matrix[:, ic]

            # Convert the rank function to an adjacency matrix and then to a hashable object
            out[ic] = AdjacencyMatrix.from_rank_function(rank_function).tohashable()

        return out.view(cls)

    def to_rank_function_matrix(self):
        """
        Use the Borda count to rank elements, return the ranks arranged in columns.
        out[i, j] is the rank of alternative (method) i according to voter (experimental condition) j.
        rf.to_numpy(dtype=int) == self.to_rank_function_matrix()
        """

        self._get_na_nv()

        out = np.zeros((self.na, self.nv), dtype=int)
        for iv, amv in enumerate(self):  # index of voter, adjacency matrix of voter
            out[:, iv] = np.unique(np.sum(np.frombuffer(amv, dtype=np.int8).reshape(self.na, self.na),
                                          axis=0),
                                   return_inverse=True)[1]
        return out

    def get_rank_function_matrix(self):
        """
        Set the rf attribute of self, containing the rank function representation.
        """
        try:
            self.rf
        except AttributeError:
            self.rf = self.to_rank_function_matrix()
        finally:
            return self.rf

    def set_key(self, key: Collection):
        """
        Set the key of entries. key must have the same length as self.
        Useful for advanced sampling, e.g., sampling datasets.
        Entries of key may not be unique, the idea is that to every key are associated multiple elements of self.
        """
        assert len(key) == len(self), f"Entered key has length {len(key)}, while it should have length {len(self)}"
        self.key = np.array(key)
        return self

    def get_subsamples_pair(self, subsample_size: int, seed: int, use_key: bool = False, replace: bool = False,
                            disjoint: bool = True, use_rf: bool = False):
        """

        :param seed: passed to the rng
        :type seed:
        :param use_key: if True, subsample using sample.key (instead of sampling from sample.index). subsample_size must be adjusted accordingly.
        :type use_key:
        :param replace: if True, sample with replacement. Allow repetitions within a subsample
        :type replace:
        :param disjoint: if True, the returned subsamples have disjoint keys (if use_key) or indices. Allow repetitions between subsamples.
        :type disjoint:
        :param use_rf: if True, return a rank function matrix
        :type use_rf:
        :return:
        :rtype:
        """
        try:
            max_size = len(set(self.key)) if use_key else len(self)
        except AttributeError:
            raise ValueError("The input sample has not key associated to it. Use sample.set_key to set one.")
        max_size //= 2 if disjoint else 1

        if subsample_size > max_size:
            raise ValueError(f"Size of subsamples is too large, must be at most {max_size}.")

        keys = self.key if use_key else range(len(self))

        if disjoint and replace:  # get two disjoint subsamples, sample from them with repetition
            shuffled = np.random.default_rng(seed).choice(keys, len(keys), replace=False)
            keys1 = np.random.default_rng(seed+1).choice(shuffled[:len(keys)//2], subsample_size, replace=True)
            keys2 = np.random.default_rng(seed+3).choice(shuffled[len(keys)//2:], subsample_size, replace=True)
        elif disjoint or replace:  # if disjoint, sample with no replacement and viceversa
            keys1, keys2 = np.random.default_rng(seed).choice(keys, 2*subsample_size, replace=replace).reshape(2, subsample_size)
        else:  # if not disjoint and no replacement, we need to sample twice
            keys1 = np.random.default_rng(seed).choice(keys, subsample_size, replace=replace)
            keys2 = np.random.default_rng(seed+1).choice(keys, subsample_size, replace=replace)

        mask1 = np.isin(keys, keys1)
        mask2 = np.isin(keys, keys2)

        if use_rf:
            out = self.get_rank_function_matrix()
            return out[:, mask1], out[:, mask2]
        else:
            return self[mask1], self[mask2]


def get_rankings_from_df(df: pd.DataFrame, factors: Iterable, alternatives: AnyStr, target: AnyStr, lower_is_better=True,
                         impute_missing=True, verbose=False) -> pd.DataFrame:
    """
        Compute a ranking of 'alternatives' for each combination of 'factors', according to 'target'.
        Set lower_is_better = False if 'target is a score, to True if it is a loss or a rank.
        If impute_missing == True, the empty ranks are imputed.
    """

    if not set(factors).issubset(df.columns):
        raise ValueError("factors must be an iterable of columns of df.")
    if alternatives not in df.columns:
        raise ValueError("alternatives must be a column of df.")
    if target not in df.columns:
        raise ValueError("target must be a column of df.")

    rankings = {}
    iterator = df.groupby(factors).groups.items()
    if verbose:
        iterator = tqdm(list(iterator))
    for group, indices in iterator:
        score = df.iloc[indices].set_index(alternatives)[target]
        rankings[group] = rlu.score2rf(score, lower_is_better=lower_is_better, impute_missing=impute_missing)

    return pd.DataFrame.from_dict(rankings, orient="index").T  # for whatever reason, this seems to be more stable


def universe_untied_rankings(na: int) -> UniverseAM[AdjacencyMatrix]:
    """
    :param na: number of alternatives, i.e., items ranked
    """
    # all possible untied rank functions
    return UniverseAM(AdjacencyMatrix.from_rank_function(rf) for rf in permutations(range(na)))


# Function for generating Rankings without ties
def generate_rankings_without_ties(na: int):
    total_orders = []
    elements_list = list(range(1, na + 1))

    for perm in permutations(elements_list):
        matrix_size = na
        adj_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

        # Map elements to indices
        element_to_index = {element: idx for idx, element in enumerate(elements_list)}

        # Fill the matrix based on the permutation
        for i in range(matrix_size):
            for j in range(i, matrix_size):
                adj_matrix[element_to_index[perm[i]], element_to_index[perm[j]]] = 1

        # Add reflexivity
        np.fill_diagonal(adj_matrix, 1)

        total_orders.append(adj_matrix)

    assert (math.factorial(na) == len(total_orders))

    return total_orders


# TODO: Function for generating Rankings with ties
def generate_rankings_with_ties(na: int):
    # could be useful: https://stackoverflow.com/questions/41210142/get-all-permutations-of-a-numpy-array
    pass


# TODO: Function for generating Rankings with ties & fails
def generate_rankings_with_ties_and_failures(na: int):
    pass


def calculate_mmd_samples(rep, num_rankings, num_samples, distribution_class, mmd_function, seed_range=(0, 10000)):
    """
    Calculate MMD for samples generated from a specified distribution class with given parameters.

    Args:
        rep (int): Number of repetitions for MMD calculation.
        num_rankings (int): Number of rankings to generate without ties.
        num_samples (int): Number of samples to generate in each distribution.
        distribution_class: Class to be used for generating distributions.
        mmd_function: Function used to calculate MMD.
        kernel_function: Kernel function to be used in MMD calculation.
        seed_range (tuple): Range of seeds for random number generation.

    Returns:
        list: A list containing the MMD universe for each repetition.
    """
    mmd_values = []
    rng = np.random.default_rng()

    rankings = generate_rankings_without_ties(num_rankings)

    for _ in range(rep):
        seed1, seed2 = rng.integers(*seed_range, size=2)

        samples1 = distribution_class(rankings).sample(num_samples, seed=seed1)
        samples2 = distribution_class(rankings).sample(num_samples, seed=seed2)

        mmd_values.append(mmd_function(samples1, samples2, mallows_kernel))

    return mmd_values


def calculate_ranking_appearances(rankings, samples):
    samples_as_bytes = [sample.tobytes() for sample in samples]
    ranking_counts = Counter(samples_as_bytes)
    return [ranking_counts.get(ranking.tobytes(), 0) for ranking in rankings]
