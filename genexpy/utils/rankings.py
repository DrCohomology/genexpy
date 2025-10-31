"""
Utility module for handling relations in ranking systems.

This module includes classes for representing rankings as adjacency matrices,
as well as utilities for converting between rankings and matrices, and for
managing collections of adjacency matrices. It provides a framework for
working with rank vectors, adjacency matrices, and their operations.
"""

import numpy as np
import pandas as pd

from collections import Counter
from collections.abc import Collection
from tqdm import tqdm
from typing import AnyStr, Iterable, Union

from genexpy.utils import relations as rlu


class AdjacencyMatrix(np.ndarray):
    """
    Class to represent a ranking as an adjacency matrix.

    The adjacency matrix M is constructed such that M[i, j] = int(R[i] <= R[j],
    where R is the ranking.
    AdjacencyMatrix objects are hashable and can therefore be used as keys for
    dictionaries.

    Parameters
    ----------
    input_array : np.ndarray
        A 2D square array representing the adjacency matrix.

    Methods
    -------
    zero(na: int) -> AdjacencyMatrix
        Creates an adjacency matrix for the zero-ranking (everything tied), of size na x na.
    from_rank_vector(rv: Iterable) -> AdjacencyMatrix
        Constructs an adjacency matrix from a rank vector.
    from_bytes(bytestring: bytes, shape: Iterable[int]) -> AdjacencyMatrix
        Creates an adjacency matrix from a bytestring.
    tohashable() -> bytes
        Converts the adjacency matrix to a hashable byte representation.
    get_ntiers() -> int
        Returns the number of unique ranks (tiers) in the adjacency matrix.
    """

    __slots__ = ()

    def __new__(cls, input_array):
        assert len(input_array.shape) == 2, "Wrong number of dimensions."
        assert input_array.shape[0] == input_array.shape[1], "An adjacency matrix is always square."
        assert np.all(input_array == input_array.astype(bool).astype(int)), "Matrix is not boolean."
        return np.asarray(input_array).view(cls)

    @classmethod
    def zero(cls, na):
        """Creates a zeroed adjacency matrix of size na x na."""
        return np.ones((na, na)).view(cls)

    @classmethod
    def from_rank_vector(cls, rv: Iterable) -> "AdjacencyMatrix":
        """
        Constructs an adjacency matrix from a rank vector.

        Parameters
        ----------
        rv : Iterable
            A rank vector representing the order of alternatives.

        Returns
        -------
        AdjacencyMatrix
            An adjacency matrix representation of the rank vector.
        """
        return np.array([[ri <= rj for rj in rv] for ri in rv]).astype(int).view(cls)

    @classmethod
    def from_bytes(cls, bytestring: bytes, shape: Iterable[int]) -> "AdjacencyMatrix":
        """
        Creates an adjacency matrix from a bytestring.

        Parameters
        ----------
        bytestring : bytes
            A bytestring representation of the adjacency matrix.
        shape : Iterable[int]
            The shape of the matrix.

        Returns
        -------
        AdjacencyMatrix
            An adjacency matrix constructed from the bytestring.
        """
        return np.frombuffer(bytestring, dtype=np.int8).reshape(shape).view(cls)

    def tohashable(self) -> bytes:
        """Converts the adjacency matrix to a hashable byte representation."""
        return self.astype(np.int8).tobytes()

    def get_ntiers(self) -> int:
        """
        Returns the number of unique ranks (tiers) in the adjacency matrix.

        Returns
        -------
        int
            The number of unique ranks in the adjacency matrix.
        """
        return len(set(np.sum(self, axis=1)))

    def __hash__(self) -> int:
        """Computes the hash of the adjacency matrix based on its byte representation."""
        return hash(self.tohashable())


class UniverseAM(np.ndarray):
    """
    Class representing a set of AdjacencyMatrix objects.

    This class stores binary encodings of adjacency matrices in an array-like
    structure. The dtype is set to object to preserve the integrity of the
    adjacency matrices' representations.

    Parameters
    ----------
    input_iter : Iterable
        An iterable of AdjacencyMatrix objects or their hashable representations.

    Methods
    -------
    to_adjmat_array(shape: Iterable[int]) -> np.ndarray
        Converts the support of hashes back to an array of AdjacencyMatrix objects.
    merge(other: UniverseAM) -> UniverseAM
        Merges with another UniverseAM instance, retaining unique entries.
    """

    def __new__(cls, input_iter: Iterable):
        try:
            return np.asarray([x.tohashable() for x in input_iter], dtype=object).view(cls)
        except AttributeError:
            if isinstance(input_iter, np.ndarray) or isinstance(input_iter, UniverseAM):
                return input_iter.view(cls)
            raise ValueError("Invalid input to UniverseAM.")

    def to_adjmat_array(self, shape: Iterable[int]) -> np.ndarray:
        """
        Converts the binary encodings back to an array of AdjacencyMatrix objects.

        Parameters
        ----------
        shape : Iterable[int]
            The shape of the adjacency matrices to be reconstructed.

        Returns
        -------
        np.ndarray
            An array of AdjacencyMatrix objects.
        """
        return np.array([AdjacencyMatrix.from_bytes(x, shape) for x in self])

    def __contains__(self, bstring: bytes) -> bool:
        """
        Checks if a bytestring representation is contained in the support.

        Parameters
        ----------
        bstring : bytes
            The bytestring to check for presence in the support.

        Returns
        -------
        bool
            True if the bytestring is present, False otherwise.
        """
        return np.any(np.isin(self, bstring))

    def _get_na_nv(self) -> None:
        """Determines the number of alternatives (methods) and sets attributes."""
        na = np.sqrt(len(self[0]))
        assert na == int(na), "Wrong length"
        self.na = int(na)
        self.nv = len(self)

    def get_na(self) -> int:
        """
        Returns the number of alternatives (methods).

        Returns
        -------
        int
            The number of alternatives in the support.
        """
        self._get_na_nv()
        return self.na

    def merge(self, other: 'UniverseAM') -> 'UniverseAM':
        """
        Merges with another UniverseAM instance, retaining unique entries.

        Parameters
        ----------
        other : UniverseAM
            The other UniverseAM instance to merge with.

        Returns
        -------
        UniverseAM
            A new UniverseAM instance containing unique entries from both.
        """
        return np.unique(np.append(self, other)).view(UniverseAM)


class SampleAM(UniverseAM):
    """
    Class representing a sample of adjacency matrices.

    This class extends UniverseAM to include additional methods for working
    with samples of adjacency matrices, including conversion from rank vectors
    and storing multiple rank vectors in a rank matrix.

    Attributes
    ----------
    rv : np.ndarray, optional
        The rank vector matrix representation of the sample.
    ntiers : int, optional
        The number of tiers per ranking in the sample.

    Methods
    -------
    from_rank_vector_dataframe(rv: pd.DataFrame) -> SampleAM
        Constructs a SampleAM from a DataFrame of rank vectors.
    from_rank_vector_matrix(rv_matrix: np.ndarray) -> SampleAM
        Converts a rank function matrix into a SampleAM object.
    to_rank_vector_matrix() -> np.ndarray
        Returns a matrix of ranks arranged by method and voter.
    get_rank_vector_matrix() -> np.ndarray
        Retrieves or computes the rank vector matrix representation.
    set_key(key: Collection) -> None
        Sets a key for entries in the sample.
    get_subsamples_pair(subsample_size: int, seed: int, use_key: bool = False, replace: bool = False,
                            disjoint: bool = True) -> tuple[SampleAM, SampleAM]:
        Draws two subsamples from the sample.
    get_subsample(subsample_size: int, seed: int, use_key: bool = False, replace: bool = False) -> SampleAM:
        Draws a single subsample from the sample.
    get_support_pmf() -> tuple[SampleAM, np.ndarray]:
        Returns the support of unique rankings and their probability mass function (PMF).
    """

    rv = None  # rank vector matrix representation of the sample
    ntiers = None  # number of tiers per ranking in the sample

    def __new__(cls, *args, **kwargs):
        """Creates a new instance of SampleAM."""
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def from_rank_vector_dataframe(cls, rv: pd.DataFrame) -> 'SampleAM':
        """
        Constructs a SampleAM from a DataFrame of rank vectors.

        Parameters
        ----------
        rv : pd.DataFrame
            DataFrame where each row represents an alternative and each column
            represents a voter. For instance, in benchmarking, a voter is an
            experimental condition. every experimental condition produces a
            ranking of the benchmarked alternatives.

        Returns
        -------
        SampleAM
            A SampleAM instance.
        """
        out = np.empty_like(rv.columns)
        for ic, col in enumerate(rv.columns):
            out[ic] = AdjacencyMatrix.from_rank_vector(rv[col]).tohashable()
        return out.view(cls)

    @classmethod
    def from_rank_vector_matrix(cls, rv_matrix: np.ndarray) -> 'SampleAM':
        """
        Converts a rank function matrix into a SampleAM object.

        Parameters
        ----------
        rv_matrix : np.ndarray
            A matrix where each row represents an alternative and each column
            represents an experimental condition or voter.

        Returns
        -------
        SampleAM
            A SampleAM instance constructed from the rank function matrix.
        """
        out = np.empty(rv_matrix.shape[1], dtype=object)  # Assuming rv_matrix.shape[1] is the number of columns/voters

        # Iterate through each experimental condition/voter
        for ic in range(rv_matrix.shape[1]):
            # Extract the rank function for the current column
            rank_vector = rv_matrix[:, ic]

            # Convert the rank function to an adjacency matrix and then to a hashable object
            out[ic] = AdjacencyMatrix.from_rank_vector(rank_vector).tohashable()

        return out.view(cls)

    def to_rank_vector_matrix(self) -> np.ndarray:
        """
        Returns a matrix of ranks arranged by method and voter.

        The output matrix contains ranks such that out[i, j] is the rank of
        alternative (method) i according to voter (experimental condition) j.

        Returns
        -------
        np.ndarray
            A matrix of ranks corresponding to the methods and voters.
        """
        self._get_na_nv()

        out = np.zeros((self.na, self.nv), dtype=int)
        for iv, amv in enumerate(self):  # index of voter, adjacency matrix of voter
            out[:, iv] = np.unique(np.sum(np.frombuffer(amv, dtype=np.int8).reshape(self.na, self.na),
                                          axis=0),
                                   return_inverse=True)[1]
        return out

    def get_rank_vector_matrix(self) -> np.ndarray:
        """
        Retrieves or computes the rank vector matrix representation.

        If the rank vector matrix has not been computed, it computes it and
        sets the rv attribute.

        Returns
        -------
        np.ndarray
            The rank vector matrix representation of the sample.
        """
        if self.rv is None:
            self.rv = self.to_rank_vector_matrix()
        return self.rv

    def set_key(self, key: Collection) -> 'SampleAM':
        """
        Set the key of entries. key must have the same length as self.
        Useful for advanced sampling, e.g., sampling datasets.
        Entries of key may not be unique, the idea is that to every key are associated multiple elements of self.

        Parameters
        ----------
        key : Collection
            A collection representing the key for the sample entries.

        Returns
        -------
        SampleAM
            The updated SampleAM instance with the set key.
        """
        assert len(key) == len(self), f"Entered key has length {len(key)}, while it should have length {len(self)}"
        self.key = np.array(key)
        return self

    def get_subsamples_pair(self, subsample_size: int, seed: int, use_key: bool = False, replace: bool = False,
                            disjoint: bool = True) -> tuple['SampleAM', 'SampleAM']:
        """
        Draws two subsamples from the sample.

        Parameters
        ----------
        subsample_size : int
            The size of each subsample.
        seed : int
            The random seed to use for subsampling.
        use_key : bool, optional
            If True, subsample using sample.key (instead of sampling from sample.index).
            subsample_size must be adjusted accordingly. The default is False.
        replace : bool, optional
            If True, sample with replacement. Allow repetitions within a subsample.
            The default is False.
        disjoint : bool, optional
            If True, the returned subsamples have disjoint keys (if use_key) or indices.
            Allow repetitions between subsamples. The default is True.

        Returns
        -------
        tuple[SampleAM, SampleAM]
            A tuple containing the two subsamples.

        Raises
        ------
        ValueError
            If use_key is True, or if the subsample size is too large.
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

        rng = np.random.default_rng(seed)

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


    def get_subsample(self, subsample_size: int, seed: int, use_key: bool = False, replace: bool = False) -> 'SampleAM':
        """
        Get a subsample of self.
        use_key is deprecated and not supported anymore.

        Parameters
        ----------
        subsample_size : int
            The size of the subsample.
        seed : int
            The random seed to use for subsampling.
        use_key : bool, optional
            If True, subsample using sample.key (instead of sampling from sample.index).
            The default is False.
        replace : bool, optional
            If True, sample with replacement. The default is False.

        Returns
        -------
        SampleAM
            A subsample of the original sample.

        Raises
        ------
        ValueError
            If use_key is True, or if the subsample size is too large.
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

    def get_support_pmf(self) -> tuple['SampleAM', np.ndarray]:
        """
        Returns the support of unique rankings and their probability mass function (PMF).

        Returns
        -------
        tuple[SampleAM, np.ndarray]
            A tuple containing the support of unique rankings and their PMF.
        """
        counter = Counter(self)
        support = SampleAM(np.array(list(counter.keys())))
        pmf = np.array(list(counter.values()), dtype=float)
        return support, pmf / np.sum(pmf)

    def get_ntiers(self):
        """
        Number of tiers of the rankings in the sample.
        Assumes that the ranks are integers and compact. I.e., ranking 0133 is not valid, 0122 is.
        """
        if self.ntiers is None:
            self.get_rank_vector_matrix()
            self.ntiers = np.max(self.rv, axis=0) - np.min(self.rv, axis=0)
        return self.ntiers

    def partition_with_ntiers(self):
        """
        Split self into a tuple of arrays. The entries of each array are ranks, and the corresponding rankings have the
            same number of tiers.
        Return a dictionary {ntier: column_vector_rankings}
        """
        return {ntier: self[self.get_ntiers() == ntier]
                for ntier in self.get_ntiers()}

    def append(self, other):
        return np.append(self, other).view(SampleAM)

    def _multisample_disjoint_replace(self, rep: int, n: int, rng: np.random.Generator):
        """
        Get 'rep' pairs of subsamples of size 'n', sampled with replacement from disjoint subsamples of 'self'.
        'self' has shape (N. ).

        Algorithm:
        1. Get rep copies of sample (rep, N).
        2. Shuffle each row independently.
        3. Split every row (roughly) in half and sample from each half independently.
        """
        N = len(self)
        samples = np.broadcast_to(np.expand_dims(self, axis=0), (rep, N))  # (rep, N)
        shuffled = rng.permuted(samples, axis=1)
        subs1 = np.array([rng.choice(sub, n, replace=True) for sub in shuffled[:, :N // 2]])  # (rep, n)
        subs2 = np.array([rng.choice(sub, n, replace=True) for sub in shuffled[:, N // 2:]])  # (rep, n)

        return subs1, subs2

    def _multisample_disjoint_not_replace(self, rep: int, n: int, rng: np.random.Generator):
        """
        Get 'rep' pairs of subsamples of size 'n', sampled with replacement from disjoint subsamples of 'self'.
        'self' has shape (N. ).

        Algorithm:
        1. Get rep copies of self (rep, N).
        2. Shuffle each row independently.
        3. Split every row (roughly) in half and sample from each half independently.
        """
        N = len(self)
        samples = np.broadcast_to(np.expand_dims(self, axis=0), (rep, N))  # (rep, N)
        shuffled = rng.permuted(samples, axis=1)
        subs1 = np.array([rng.choice(sub, n, replace=False) for sub in shuffled[:, :N // 2]])  # (rep, n)
        subs2 = np.array([rng.choice(sub, n, replace=False) for sub in shuffled[:, N // 2:]])  # (rep, n)

        return subs1, subs2

    def _multisample_not_disjoint_replace(self, rep: int, n: int, rng: np.random.Generator):
        """
        Get 'rep' pairs of samples of size 'n', sampled with replacement from 'sample'.
        'sample' has shape (N. ).

        Algorithm:
        1. Get rep copies of sample (rep, N).
        2. Get a sample of size 2n from each row independently.
        3. Split the rows in half.
        """
        N = len(self)
        samples = np.broadcast_to(np.expand_dims(self, axis=0), (rep, N))  # (rep, N)
        tmp = np.array([rng.choice(sub, 2 * n, replace=True) for sub in samples])  # (rep, 2*n)
        subs1 = tmp[:, :n]
        subs2 = tmp[:, n:]

        return subs1, subs2

    def _multisample_not_disjoint_not_replace(self, rep: int, n: int, rng: np.random.Generator):
        """
        Get 'rep' pairs of samples of size 'n', sampled with replacement from 'self'.
        'sample' has shape (N. ).

        Algorithm:
        1. Get rep copies of self (rep, N).
        2. Get a sample without replacement of size 2n from each row independently.
        3. Split the rows in half.
        """
        N = len(self)
        samples = np.broadcast_to(np.expand_dims(self, axis=0), (rep, N))  # (rep, N)
        subs1 = np.array([rng.choice(sub, n, replace=False) for sub in samples])  # (rep, n)
        subs2 = np.array([rng.choice(sub, n, replace=False) for sub in samples])  # (rep, n)

        return MultiSampleAM(subs1), MultiSampleAM(subs2)

    def get_multisample_pair(self, subsample_size: int, rep: int, seed: int, disjoint: bool = True,
                             replace: bool = False):
        """
        Get 'rep' pairs of subsamples of size 'n', sampled from 'self' (which has shape (N, )).
        If disjoint is True, the subsampled are sampled form two disjoint pools of indices of 'self'.
        If replace is True, the sampling is with replacement.
        """

        rng = np.random.default_rng(seed)

        match (disjoint, replace):
            case (True, True):
                return self._multisample_disjoint_replace(rep=rep, n=subsample_size, rng=rng)
            case (True, False):
                return self._multisample_disjoint_not_replace(rep=rep, n=subsample_size, rng=rng)
            case (False, True):
                return self._multisample_not_disjoint_replace(rep=rep, n=subsample_size, rng=rng)
            case (False, False):
                return self._multisample_not_disjoint_not_replace(rep=rep, n=subsample_size, rng=rng)


class MultiSampleAM(np.ndarray):
    """
    A sample of samples (a 2d sample).

    This class represents a collection of samples, where each sample is itself a
    collection of adjacency matrices. It provides methods for converting between
    different representations of the multi-sample, such as rank vectors and
    adjacency matrices.

    Attributes
    ----------
    rep : int
        The number of samples in the multi-sample.
    na : int
        The number of alternatives in each sample.
    n : int
        The size of each sample.

    Methods
    -------
    to_rank_vectors() -> np.ndarray
        Converts the multi-sample to a representation of rank vectors.
    to_adjacency_matrices(na: int) -> np.ndarray
        Converts the multi-sample to a representation of adjacency matrices.
    """

    def __new__(cls, input_iter: Iterable):
        """Creates a new instance of MultiSampleAM."""
        return np.asarray(input_iter).view(cls)

    def to_rank_vectors(self) -> np.ndarray:
        """
        Converts the multi-sample to a representation of rank vectors.

        Returns
        -------
        np.ndarray
            A 3D array of shape (rep, na, n) representing the rank vectors,
            where rep is the number of samples, na is the number of alternatives,
            and n is the size of each sample.
        """
        return np.array([SampleAM(sample).to_rank_vector_matrix() for sample in self])      # (rep, na, n)

    def to_adjacency_matrices(self, na: int) -> np.ndarray:
        """
        Converts the multi-sample to a representation of adjacency matrices.

        Parameters
        ----------
        na : int
            The number of alternatives in each adjacency matrix.

        Returns
        -------
        np.ndarray
            A 4D array of shape (rep, n, na, na) representing the adjacency matrices,
            where rep is the number of samples, n is the size of each sample,
            and na is the number of alternatives.
        """
        return np.array([[AdjacencyMatrix.from_bytes(r, shape=(na, na)) for r in sample] for sample in self])    # (rep, n, na, na)

    def get_pmfs_df(self, support: UniverseAM = None) -> pd.DataFrame:
        """
        Create a dataframe. Index: rankings. Columns: samples in 'ms'.

        Parameters
        ----------

        Returns
        -------
        a pd.DataFrame

        """
        # If support is None, it is ignored. If it is not, the index of the output df is guaranteed to have
        # `support` as a subset
        tmps = [pd.Series(index=support, name="support_tmp")]
        for i, s in enumerate(self):
            support_lcl, pmf = SampleAM(s).get_support_pmf()
            if support is not None and not set(support_lcl).issubset(support):
                raise ValueError("There are rankings in the sample that are not contained in self.support (which is "
                                 "not None)")

            tmps.append(pd.Series(pmf, index=support_lcl))

        return pd.concat(tmps, axis=1, ignore_index=False).drop(columns="support_tmp").fillna(0)


def get_matrix_from_df(df: pd.DataFrame, factors: Iterable, alternatives: AnyStr, target: AnyStr,
                       impute_missing=True, tol_missing_indices: float = 0.2,
                       tol_missing_columns: float = 0.2,
                       get_rankings: bool = True, lower_is_better: bool = True,
                       as_numpy: bool=False, verbose: bool = True) -> Union[pd.DataFrame, np.ndarray]:
    """
    Computes a ranking of 'alternatives' for each combination of 'factors', according to 'target'.

    This function groups the DataFrame by the specified factors and then ranks the
    alternatives within each group based on the target column.

    Parameters
    ----------
    as_numpy :
    get_rankings :
    lower_is_better :
    df : pd.DataFrame
        The DataFrame containing the data.
    factors : Iterable
        An iterable of column names to group the DataFrame by.
    alternatives : AnyStr
        The name of the column containing the alternatives to be ranked.
    target : AnyStr
        The name of the column containing the values to rank by.
    impute_missing : bool, optional
        Whether to impute missing values in the rankings. The default is True.
    tol_missing_indices: float = 0
        maximum allowed fraction of missing indices for a column to be kept
    tol_missing_columns: float = 0
        maximum allowed fraction of missing columns for an index to be kept

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the rankings of alternatives for each combination of factors.

    Raises
    ------
    ValueError
        If any of the factors, alternatives, or target columns are not present in the DataFrame.
    """

    if not set(factors).issubset(df.columns):
        raise ValueError("factors must be an iterable of columns of df.")
    if alternatives not in df.columns:
        raise ValueError("alternatives must be a column of df.")
    if target not in df.columns:
        raise ValueError("target must be a column of df.")

    out = df.reset_index(drop=True).pivot(index=alternatives, columns=factors, values=target)
    original_shape = out.shape

    # filter out columns
    out = out.loc[:, out.isna().mean(axis=0) <= tol_missing_indices]
    out = out.loc[out.isna().mean(axis=1) <= tol_missing_columns, :]

    if verbose:
        print(f"[INFO] Kept {out.shape[0]} / {original_shape[0]} indices (alternatives) and "
              f"{out.shape[1]} / {original_shape[1]} columns (conditions).")

    if impute_missing:
        out = out.fillna(0)

    if get_rankings:
        out = out.apply(lambda x: rlu.score2rv(x, lower_is_better=lower_is_better))

    if as_numpy:
        return out.to_numpy()

    return out