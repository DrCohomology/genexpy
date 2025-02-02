import math
import numpy as np
import pandas as pd

from collections.abc import Collection
from itertools import permutations
from tqdm import tqdm
from typing import AnyStr, Iterable

from . import relation_utils as rlu


class AdjacencyMatrix(np.ndarray):
    """
    Store a ranking as an adjacency matrix.
    M[i, j] = int(R[i] <= R[j])
    """
    __slots__ = ()
    def __new__(cls, input_array):
        assert len(input_array.shape) == 2, "Wrong number of dimensions."
        assert input_array.shape[0] == input_array.shape[1], "An adjacency matrix is always square."
        assert np.all(input_array == input_array.astype(bool).astype(int)), "Matrix is not boolean."
        return np.asarray(input_array).view(cls)

    @classmethod
    def from_rank_function(cls, rv: Iterable):
        """
        a rank function maps an alternative into its rank
        """
        return np.array([[ri <= rj for rj in rv] for ri in rv]).astype(int).view(cls)

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
    def from_rank_function_dataframe(cls, rv: pd.DataFrame):
        """
        Each row of rv is an alternative.
        Each column of rv is an experimental condition/ voter.
        """
        out = np.empty_like(rv.columns)
        for ic, col in enumerate(rv.columns):
            out[ic] = AdjacencyMatrix.from_rank_function(rv[col]).tohashable()
        return out.view(cls)

    @classmethod
    def from_rank_function_matrix(cls, rv_matrix):
        """
        Convert a rank function matrix into a SampleAM object.
        Each row of rv_matrix is an alternative.
        Each column of rv_matrix is an experimental condition/ voter.
        """
        # Initialize an array to store hashable representations of adjacency matrices
        out = np.empty(rv_matrix.shape[1], dtype=object)  # Assuming rv_matrix.shape[1] is the number of columns/voters

        # Iterate through each experimental condition/voter
        for ic in range(rv_matrix.shape[1]):
            # Extract the rank function for the current column
            rank_function = rv_matrix[:, ic]

            # Convert the rank function to an adjacency matrix and then to a hashable object
            out[ic] = AdjacencyMatrix.from_rank_function(rank_function).tohashable()

        return out.view(cls)

    def to_rank_function_matrix(self):
        """
        Use the Borda count to rank elements, return the ranks arranged in columns.
        out[i, j] is the rank of alternative (method) i according to voter (experimental condition) j.
        rv.to_numpy(dtype=int) == self.to_rank_function_matrix()
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
        Set the rv attribute of self, containing the rank function representation.
        """
        try:
            self.rv
        except AttributeError:
            self.rv = self.to_rank_function_matrix()
        finally:
            return self.rv

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
                            disjoint: bool = True):
        """

        :param seed: passed to the rng
        :type seed:
        :param use_key: if True, subsample using sample.key (instead of sampling from sample.index). subsample_size must be adjusted accordingly.
        :type use_key:
        :param replace: if True, sample with replacement. Allow repetitions within a subsample
        :type replace:
        :param disjoint: if True, the returned subsamples have disjoint keys (if use_key) or indices. Allow repetitions between subsamples.
        :type disjoint:
        :return:
        :rtype:
        """

        if use_key:
            raise ValueError("use_key = True is not accepted anymore.")

        try:
            max_size = len(set(self.key)) if use_key else len(self)
        except AttributeError:
            raise ValueError("The input sample has not key associated to it. Use sample.set_key to set one.")
        max_size //= 2 if disjoint else 1

        if not replace and subsample_size > max_size:
            raise ValueError(f"Size of subsamples is too large, must be at most {max_size}.")

        rng = np.random.RandomState(seed)

        if disjoint and replace:  # get two disjoint subsamples, then samples from them
            shuffled = rng.choice(self, len(self), replace=False)
            out1 = rng.choice(shuffled[:len(self) // 2], subsample_size, replace=True)
            out2 = rng.choice(shuffled[len(self) // 2:], subsample_size, replace=True)
        elif disjoint or replace:  # implies replace = not disjoint
            out1, out2 = rng.choice(self, 2*subsample_size, replace=replace).reshape(2, subsample_size)
        else:  # if not disjoint and no replacement, we just sample twice
            out1 = rng.choice(self, subsample_size, replace=False)
            out2 = rng.choice(self, subsample_size, replace=False)

        return SampleAM(out1), SampleAM(out2)

    def get_subsample(self, subsample_size: int, seed: int, use_key: bool = False, replace: bool = False):
        """
        Get a subsample of self.
        use_key is deprecated and not supported anymore.
        """

        if use_key:
            raise ValueError("use_key = True is not accepted anymore.")

        try:
            max_size = len(set(self.key)) if use_key else len(self)
        except AttributeError:
            raise ValueError("The input sample has not key associated to it. Use sample.set_key to set one.")

        if not replace and subsample_size > max_size:
            raise ValueError(f"Size of subsamples is too large, must be at most {max_size}.")

        return SampleAM(np.random.default_rng(seed).choice(self, subsample_size, replace=replace))

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
        rankings[group] = rlu.score2rv(score, lower_is_better=lower_is_better, impute_missing=impute_missing)

    return pd.DataFrame.from_dict(rankings, orient="index").T  # for whatever reason, this seems to be more stable


def universe_untied_rankings(na: int) -> UniverseAM[AdjacencyMatrix]:
    """
    :param na: number of alternatives, i.e., items ranked
    """
    # all possible untied rank functions
    return UniverseAM(AdjacencyMatrix.from_rank_function(rv) for rv in permutations(range(na)))


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