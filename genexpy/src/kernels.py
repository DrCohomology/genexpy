from itertools import product

import numpy as np
import pandas as pd

from numba import njit
from typing import Literal, TypeAlias, Union

from . import rankings_utils as ru


RankVector: TypeAlias = np.ndarray[int]
RankByte: TypeAlias = bytes
Ranking: TypeAlias = Union[RankVector, RankByte]



class Kernel:

    vectorized_input_format: Literal["adjmat", "vector"] = None

    def __init__(self, universe: ru.UniverseAM = None, **kwargs) -> None:
        self.universe = universe  # universe of rankings
        self.K = None  # gram matrix of the universe

    def set_universe(self, universe: ru.UniverseAM):
        self.universe = universe
        self.K = None

    def get_eps(self, delta, na: int = None):
        pass

    def _validate_parameters(self):
        pass

    @staticmethod
    def _validate_inputs(x1: Ranking, x2:Ranking):
        if len(x1) != len(x2):
            raise ValueError("Ranking dimensions do not match")
        if isinstance(x1, RankByte):
            if np.sqrt(len(x1)) != int(np.sqrt(len(x1))):
                raise ValueError(f"The input bytestring has length {len(x1)} and is not a square (adjacency) matrix.")

    @staticmethod
    def _validate_vectorized_inputs_rv(rv1: RankVector, rv2: RankVector):
        if rv1.shape != rv2.shape:
            raise ValueError("The two rank matrices' dimensions do not match.")
        elif len(rv1.shape) != 2:
            raise ValueError(f"The shape of the input should be that of a valid RankVector, i.e., (num_alternatives, num_voters). "
                             f"Object of shape {rv1.shape} is not a valid RankVector.")

    @staticmethod
    def _validate_vectorized_inputs_ams(ams1: ru.AdjacencyMatrix, ams2: ru.AdjacencyMatrix):
        if ams1.shape != ams2.shape:
            raise ValueError("The two adjacency matrices' dimensions do not match.")
        elif ams1.shape[1] != ams1.shape[2]:
            raise ValueError(f"Input with shape {ams1.shape} is not an array of adjacency matrices (shape[1] and "
                             f"shapoe[2] should coincide).")
        elif len(ams1.shape) != 3:
            raise ValueError(f"The shape of the input should be that of a valid AdjacencyMatrix, i.e., (num_voters, num_alternatives, num_alternatives). "
                             f"Object of shape {ams1.shape} is not a valid AdjacencyMatrix.")

    def _instantiate_parameters(self, *args, **kwargs):
        pass

    def __call__(self, x1: Ranking, x2: Ranking, use_rv: bool = True) -> float:
        """
        Computes the Mallows kernel between two rankings, which is based on the difference in their rankings adjusted by a
        kernel bandwidth parameter nu.

        Parameters:
        - x1 (Ranking): The first ranking as a RankVector or RankByte.
        - x2 (Ranking): The second ranking as a RankVector or RankByte.
        - nu (float, 'auto'): The decay parameter for the kernel. If 'auto', it adjusts based on the length of the rankings.
        - use_rv (bool): Determines whether to use the rank vector or byte representation for the calculation.

        Returns:
        - float: The computed Mallows kernel value.

        Raises:
        - ValueError: If the rankings do not have the same number of alternatives.
        """
        self._validate_inputs(x1, x2)
        self._instantiate_parameters(na=len(x1) if use_rv else np.sqrt(len(x1)))

        return self._rv(x1, x2) if use_rv else self._bytes(x1, x2)

    def _bytes(self, b1: RankByte, b2: RankByte) -> float:
        pass

    def _rv(self, r1: RankVector, r2: RankVector) -> float:
        pass

    def _gram_matrix_naive(self, s1: ru.SampleAM, s2: ru.SampleAM, use_rv: bool = True) -> np.ndarray[float]:
        """
        Computes the Gram matrix between two samples of rankings, where each entry in the matrix represents the kernel
        similarity between the rankings from each sample.

        Parameters:
        - sample1 (SampleAM): The first sample of rankings.
        - sample2 (SampleAM): The second sample of rankings.
        - use_rv (bool): If True, converts the rankings to rank function matrix format before processing.
        - kernel (Kernel): The kernel function to use for computing similarities.
        - **kernelargs: Additional keyword arguments for the kernel function.

        Returns:
        - np.ndarray[float]: A matrix of kernel similarities.
        """

        if use_rv:
            s1 = s1.to_rank_vector_matrix().T  # rows: voters, cols: alternatives
            s2 = s2.to_rank_vector_matrix().T  #

        out = np.zeros((len(s1), len(s2)))
        if np.equal(s1, s2).all():
            for i2, x2 in enumerate(s2):
                for i1, x1 in list(enumerate(s1))[:i2]:
                    out[i1, i2] = self(x1, x2, use_rv)
            d = np.diag([self(x, x, use_rv) for x in s1])
            return out + d + out.T
        else:
            for (i1, x1), (i2, x2) in product(enumerate(s1), enumerate(s2)):
                out[i1, i2] = self(x1, x2, use_rv)
            return out

    def _gram_matrix_scalar(self, *args) -> np.ndarray[float]:
        """
        Use as base for the vectorized gram matrix.
        Parameters
        ----------
        args :

        Returns
        -------

        """
        raise NotImplementedError()

    def _gram_matrix_vectorized(self, *args) -> np.ndarray[float]:
        raise NotImplementedError()

    def gram_matrix(self, *args) -> np.ndarray[float]:
        try:
            return self._gram_matrix_vectorized(*args)
        except NotImplementedError:
            return self._gram_matrix_naive(*args)

    def __repr__(self):
        return "Kernel"

    def __str__(self):
        return self.__repr__()

    def _convert_multisample_to_vectorized_input_format(self, ms: ru.MultiSampleAM):

        na = int(np.sqrt(len(ms[0, 0])))  # number of alternatives

        # Input checks
        if np.not_equal(na, np.sqrt(len(ms[0, 0]))):
            raise ValueError(f"The MultiSample has invalid dimension: the length of ms[0, 0] is "
                             f"{np.sqrt(len(ms[0, 0]))} and should be the square of "
                             f"the number of alternatives, but is not a perfect square.")

        match self.vectorized_input_format:
            case "adjmat":
                return ms.to_adjacency_matrices(na=na)
            case "vector":
                return ms.to_rank_vectors()
            case _:
                raise ValueError(f"Unsupported input format: {self.vectorized_input_format}")

    def _convert_sample_to_input_format(self, s: ru.SampleAM):

        na = int(np.sqrt(len(s[0])))

        # Input checks
        if np.not_equal(na, np.sqrt(len(s[0]))):
            raise ValueError(f"The first MultiSample has invalid dimension: the length of s[0] is "
                             f"{np.sqrt(len(s[0]))} and should be the square of "
                             f"the number of alternatives, but is not a perfect square.")

        match self.vectorized_input_format:
            case "adjmat":
                return s.to_adjmat_array(shape=(na, na))
            case "vector":
                return s.to_rank_vector_matrix()
            case _:
                raise ValueError(f"Unsupported input format: {self.vectorized_input_format}")

    def _mmd_distribution_naive(self, sample: ru.SampleAM, n: int, rep: int, seed: int = 0, disjoint: bool = True,
                                     replace: bool = False) -> np.ndarray[float]:

        ms1, ms2 = sample.get_multisample_pair(subsample_size=n, rep=rep, seed=seed, disjoint=disjoint, replace=replace)

        ms1 = ru.MultiSampleAM(ms1)
        ms2 = ru.MultiSampleAM(ms2)

        mmd = []
        for s1, s2 in zip(ms1, ms2):
            s1 = ru.SampleAM(s1)
            s2 = ru.SampleAM(s2)

            rv1 = s1.to_rank_vector_matrix()
            rv2 = s2.to_rank_vector_matrix()

            n = len(s1)

            Kxx = np.empty((n, n))
            diag = np.empty((n))
            for i, r1 in enumerate(rv1.T):
                diag[i] = self(r1, r1)
                for j, r2 in enumerate(rv1.T[:i]):
                    Kxx[i, j] = self(r1, r2)
            Kxx = Kxx + Kxx.T + np.diag(diag)

            Kyy = np.empty((n, n))
            diag = np.empty((n))
            for i, r1 in enumerate(rv2.T):
                diag[i] = self(r1, r1)
                for j, r2 in enumerate(rv2.T[:i]):
                    Kyy[i, j] = self(r1, r2)
            Kyy = Kyy + Kyy.T + np.diag(diag)

            Kxy = np.empty((n, n))
            for i, r1 in enumerate(rv1.T):
                for j, r2 in enumerate(rv2.T):
                    Kxy[i, j] = self(r1, r2)

            mmd.append(np.sqrt(Kxx.mean() + Kyy.mean() - 2*Kxy.mean()))

        return np.array(mmd)

    def _mmd_distribution_vectorized(self, sample: ru.SampleAM, n: int, rep: int, seed: int = 0, disjoint: bool = True,
                                     replace: bool = False) -> np.ndarray[float]:
        ms1, ms2 = sample.get_multisample_pair(subsample_size=n, rep=rep, seed=seed, disjoint=disjoint, replace=replace)

        ms1 = ru.MultiSampleAM(ms1)
        ms2 = ru.MultiSampleAM(ms2)

        x1 = self._convert_multisample_to_vectorized_input_format(ms1)
        x2 = self._convert_multisample_to_vectorized_input_format(ms2)

        Kxx = self.gram_matrix(x1, x1)
        Kxy = self.gram_matrix(x1, x2)
        Kyy = self.gram_matrix(x2, x2)

        return np.sqrt(np.abs(np.mean(Kxx, axis=(1, 2)) + np.mean(Kyy, axis=(1, 2)) - 2 * np.mean(Kxy, axis=(1, 2))))

    def _mmd_distribution_embedding(self, sample: ru.SampleAM, n: int, rep: int, seed: int = 0, disjoint: bool = True,
                                     replace: bool = False, use_cached_universe_matrix: bool = False) -> np.ndarray[float]:

        ms1, ms2 = sample.get_multisample_pair(subsample_size=n, rep=rep, seed=seed, disjoint=disjoint, replace=replace)

        ms1 = ru.MultiSampleAM(ms1)
        ms2 = ru.MultiSampleAM(ms2)

        pmf_df1 = ms1.get_pmfs_df(self.universe)
        pmf_df2 = ms2.get_pmfs_df(self.universe)

        # ms1[0] is compared to ms2[0] etc...
        # equivalently, pmf_df1.iloc[:, 0] is compared with pmf_df2.iloc[:, 0]
        alpha_df = pmf_df1 - pmf_df2
        alpha_df = alpha_df.fillna(pmf_df1).fillna(-pmf_df2)  # if a ranking does not appear in both is an NaN

        alpha = alpha_df.values

        if use_cached_universe_matrix:
            if self.universe is None:
                raise ValueError("To cache the universe matrix, self.universe must be set.")
            if self.K is not None:
                return np.sqrt(np.diag(alpha.T @ self.K @ alpha))

        # get the kernel matrix from the index of alpha (the universe)
        universe = self.universe if self.universe is not None else ru.SampleAM(alpha_df.index.values)
        x = self._convert_sample_to_input_format(universe)
        self.K = self.gram_matrix(x, x)

        return np.sqrt(np.diag(alpha.T @ self.K @ alpha))


    #TODO implement
    @staticmethod
    def _select_fastest_mmd_estimation_method() -> Literal["basic", "vectorized", "definition"]:
        return "definition"

    def mmd_distribution(self, sample: ru.SampleAM, n: int, rep: int, seed: int = 0, disjoint: bool = True,
                         replace: bool = False,
                         method: Literal["auto", "basic", "vectorized", "definition"]="auto",
                         use_cached_universe_matrix: bool = False) -> np.ndarray[float]:

        if method == "auto":
            method = self._select_fastest_mmd_estimation_method()

        match method:
            case "vectorized":
                return self._mmd_distribution_vectorized(sample, n=n, rep=rep, seed=seed,
                                                         disjoint=disjoint, replace=replace)
            case "definition":
                return self._mmd_distribution_embedding(sample, n=n, rep=rep, seed=seed,
                                                         disjoint=disjoint, replace=replace,
                                                         use_cached_universe_matrix=use_cached_universe_matrix)
            case _:
                raise ValueError(f"Invalid method {method}.")

    # TODO save the kernel matrix in cache for method="definition". we do not have to recompute it every time
    def mmd_distribution_many_n(self, sample: ru.SampleAM, nmin: int, nmax: int, step: int,
                                seed: int = 100, disjoint: bool = True, replace: bool = False, N: int = None,
                                **mmd_distribution_parms) -> pd.DataFrame:
        mmds = {n: self.mmd_distribution(sample=sample, n=n, seed=int(np.exp(seed))*n, disjoint=disjoint, replace=replace,
                                         **mmd_distribution_parms)
                for n in range(nmin, nmax, step)}

        dfmmd = pd.DataFrame(mmds).melt(var_name="n", value_name="mmd")
        dfmmd["N"] = N
        dfmmd["disjoint"] = disjoint
        dfmmd["replace"] = replace
        dfmmd["kernel"] = self.__str__()

        return dfmmd

class BordaKernel(Kernel):

    vectorized_input_format = "vector"

    def __init__(self, idx: int, nu: Union[float, Literal["auto"]] = "auto", **kwargs) -> None:
        super().__init__(**kwargs)
        # TODO let idx be the name of an alternative
        self.idx = idx
        self.nu = nu
        self._validate_parameters()

        self._gram_matrix_vectorized = np.vectorize(self._gram_matrix_scalar,
                                        signature="(na, n), (na, n) -> (n, n)", otypes=[float], excluded="self")


    def __repr__(self):
        return f"BordaKernel(nu={self.nu:.2f}, idx={self.idx})"

    # TODO: Not working if nu has been instantiated with _instantiate_parameters. easy fix is generalize the value of eps
    def get_eps(self, delta, na: int = None):
        if self.nu == "auto":
            return np.sqrt(2 * (1 - np.exp(-delta)))
        else:
            # Use nu / nu_auto.
            # delta is the difference between the fraction of dominated alternatives between two rankings
            return np.sqrt(2 * (1 - np.exp(-self.nu * na * delta)))

    def _validate_parameters(self):
        if isinstance(self.nu, str):
            if self.nu != "auto":
                raise ValueError(f"Invalid value for parameter nu={self.nu}. Accepted: positive float or 'auto'")
        elif isinstance(self.nu, float):
            if self.nu < 0:
                raise ValueError(f"Invalid value for parameter nu={self.nu}. Accepted: positive float or 'auto'")
        else:
            raise ValueError(f"Invalid value for parameter nu={self.nu}. Accepted: positive float or 'auto'")

        if isinstance(self.idx, int):
            pass
        else:
            raise ValueError(f"Invalid value for parameter idx={self.idx}. Accepted: int")

    def _validate_inputs(self, x1: Ranking, x2:Ranking):
        if len(x1) != len(x2):
            raise ValueError(f"The rankings hav different lengths {len(x1)} and {len(x2)}")
        if isinstance(x1, RankByte):
            if np.sqrt(len(x1)) != int(np.sqrt(len(x1))):
                raise ValueError(f"The input bytestring has length {len(x1)} and is not a square (adjacency) matrix.")
        if self.idx >= len(x1):
            raise ValueError(f"The idx must not exceed the length of the rankings.")

    def _instantiate_parameters(self, na):
        if self.nu == "auto":
            self.nu = 1 / na

    def _rv(self, r1: RankVector, r2: RankVector) -> float:
        return np.exp(- self.nu * np.abs(np.sum(r1 >= r1[self.idx]) - np.sum(r2 >= r2[self.idx])))

    def _bytes(self, b1: RankByte, b2: RankByte) -> float:
        raise NotImplementedError

    def _gram_matrix_scalar(self, rv1: RankVector, rv2: RankVector):
        """
        Computes the Gram matrix of the Borda kernel between two sets of rankings, represented as vectors.

        The Borda kernel is a similarity measure between two rankings,
        based on the number of elements that are ranked higher than a given element in each ranking.

        Parameters
        ----------
        rv1 : RankVector
            The first set of rankings, represented as a tensor of shape (na, n),
            where na is the number of alternatives, and n is the number of rankings.
        rv2 : RankVector
            The second set of rank vectors, with the same shape as rv1.
        idx : int
            The index of the element to compare the rankings for.
        nu : float or "auto", optional
            The scaling parameter for the kernel. If "auto", it is set to 2 / (na*(na-1)),
            where na is the number of elements. The default is "auto".

        Returns
        -------
        ndarray
            A tensor of shape (n, n) representing the Gram matrix of the Borda kernel between the two sets of
            rankings.

        Raises
        ------
        ValueError
            If the two tensors do not have the same shape.

        See Also
        --------
        RankVector : Class representing a ranking.

        Notes
        -----
        The Borda kernel is defined as:

        .. math::
            K(R_1, R_2) = exp(-nu * |d_1 - d_2|)

        where :math:`R_1` and :math:`R_2` are rank vectors, :math:`d_1` is the number of elements
        ranked higher than the element at index `idx` in :math:`R_1`, and :math:`d_2` is the
        number of elements ranked higher than the element at index `idx` in :math:`R_2`.

        Examples
        --------
        >>> from . import ranking_utils as ru
        >>> rv1 = ru.RankVector(np.array([[1, 2, 3], [3, 1, 2]]))
        >>> rv2 = ru.RankVector(np.array([[2, 1, 3], [1, 3, 2]]))
        >>> BordaKernel(idx=0, nu="auto").gram_matrix(rv1, rv2)
        array([[0.60653066, 0.36787944],n

               [0.36787944, 0.60653066]])
        """
        self._validate_vectorized_inputs_rv(rv1, rv2)
        self._instantiate_parameters(na=rv1.shape[0])

        d1 = np.sum(rv1 >= rv1[self.idx], axis=0)  # dominated
        d2 = np.sum(rv2 >= rv2[self.idx], axis=0)
        return np.exp(- self.nu * np.abs(np.expand_dims(d1, axis=1) - np.expand_dims(d2, axis=0)))


class JaccardKernel(Kernel):

    vectorized_input_format = "vector"

    def __init__(self, k:int) -> None:
        super().__init__()
        self.k = k
        self._validate_parameters()

        self._gram_matrix_vectorized = np.vectorize(self._gram_matrix_scalar,
                                        signature="(na, n), (na, n) -> (n, n)", otypes=[float], excluded="self")

    def __repr__(self):
        return f"JaccardKernel(k={self.k})"

    def get_eps(self, delta, na: int = None):
        return np.sqrt(2 * (1 - (1-delta)))

    def _validate_parameters(self):
        if not isinstance(self.k, int):
            raise ValueError(f"Invalid value for parameter k={self.k}. Accepted: int")

    def _bytes(self, b1: RankByte, b2: RankByte) -> float:
        """
        Implementation is specific for AdjacencyMatrix objects, version of 25.01.2024.
        """
        na = int(np.sqrt(len(b1)))

        topk1 = np.where(np.frombuffer(b1, dtype=np.int8).reshape((na, na)).sum(axis=1) > na - self.k)[0]
        topk2 = np.where(np.frombuffer(b2, dtype=np.int8).reshape((na, na)).sum(axis=1) > na - self.k)[0]

        return len(set(topk1).intersection(set(topk2))) / len(set(topk1).union(set(topk2)))

    def _rv(self, r1: RankVector, r2: RankVector) -> float:
        """
        Supports tied rankings as columns of the output from SampleAM.to_rank_vector_matrix().
        """
        topk1 = np.where(r1 < self.k)[0]
        topk2 = np.where(r2 < self.k)[0]

        return len(set(topk1).intersection(set(topk2))) / len(set(topk1).union(set(topk2)))

    def _gram_matrix_scalar(self, rv1: RankVector, rv2: RankVector):
        """
        Computes the Gram matrix of the Jaccard kernel between two sets of rankings.

        The Jaccard kernel is a similarity measure between two sets of rankings,
        based on the number of elements ranked within a given cutoff in each ranking.

        Parameters
        ----------
        rv1 : RankVector
            The first set of rankings, represented as a tensor of shape (na, n),
            where na is the number of alternatives, and n is the number of vectors.
        rv2 : RankVector
            The second set of rank vectors, with the same shape as rv1.
        k : int
            The cutoff value for the ranking.

        Returns
        -------
        ndarray99
            A tensor of shape (n, n) representing the Gram matrix of the Jaccard kernel between the two sets of
            rankings.

        See Also
        --------
        RankVector : Class representing a ranking.

        Notes
        -----
        The Jaccard kernel is defined as:

        .. math::
            K(R_1, R_2) = \frac{|R_1 \cap R_2|}{|R_1 \cup R_2|}

        where :math:`R_1` and :math:`R_2` are rank vectors, and :math:`R_1 \cap R_2`
        represents the set of elements ranked within the cutoff `k` in both vectors, and
        :math:`R_1 \cup R_2` represents the set of elements ranked within the cutoff `k`
        in either vector.
        """
        self._validate_vectorized_inputs_rv(rv1, rv2)
        self._instantiate_parameters(na=rv1.shape[0])

        k1 = rv1 < self.k
        k2 = rv2 < self.k
        intersection = np.logical_and(np.expand_dims(k1, 2), np.expand_dims(k2, 1)).astype(int).sum(axis=0)
        union = np.logical_or(np.expand_dims(k1, 2), np.expand_dims(k2, 1)).astype(int).sum(axis=0)
        return intersection / union


class MallowsKernel(Kernel):

    vectorized_input_format = "adjmat"

    def __init__(self, nu: Union[float, Literal["auto"]] = "auto", **kwargs) -> None:
        super().__init__(**kwargs)
        self.nu = nu
        self._validate_parameters()

        self._gram_matrix_vectorized = np.vectorize(self._gram_matrix_scalar,
                                        signature="(n, na, na), (n, na, na) -> (n, n)", otypes=[float], excluded="self")

    def __repr__(self):
        return f"MallowsKernel(nu={self.nu:.2f})"

    def get_eps(self, delta, na: int = None):
        if self.nu == "auto":
            return np.sqrt(2 * (1 - np.exp(-delta)))
        else:
            # Use nu / nu_auto
            # delta is the fraction of discordant pairs
            return np.sqrt(2 * (1 - np.exp(- self.nu * (na*(na-1)) / 2 * delta)))


    def _validate_parameters(self):
        if isinstance(self.nu, str):
            if self.nu != "auto":
                raise ValueError(f"Invalid value for parameter nu={self.nu}. Accepted: positive float or 'auto'")
        elif isinstance(self.nu, float):
            if self.nu < 0:
                raise ValueError(f"Invalid value for parameter nu={self.nu}. Accepted: positive float or 'auto'")
        else:
            raise ValueError(f"Invalid value for parameter nu={self.nu}. Accepted: positive float or 'auto'")


    def _instantiate_parameters(self, na):
        if self.nu == "auto":
            self.nu = 2 / (na*(na-1))


    def _bytes(self, b1: RankByte, b2: RankByte) -> float:
        i1 = np.frombuffer(b1, dtype=np.int8)
        i2 = np.frombuffer(b2, dtype=np.int8)
        return np.exp(- self.nu * np.sum(np.abs(i1 - i2)) / 2)


    def _rv(self, r1: RankVector, r2: RankVector) -> float:
        out = 0  # twice the number of discordant pairs ((tie, not-tie) counts as 1/2 discordant)
        for i in range(len(r1)):
            for j in range(i):
                out += np.abs(np.sign(r1[i] - r1[j]) - np.sign(r2[i] - r2[j]))
        return np.exp(- self.nu * out / 2)


    def _gram_matrix_scalar(self, ams1: ru.AdjacencyMatrix, ams2: ru.AdjacencyMatrix):
        """
        Computes the Gram matrix of the Mallows kernel between two sets of rankings,
        represented as adjacency matrices.

        The Mallows kernel is a similarity measure between two adjacency matrices,
        based on the number of discordant pairs between the two corresponding rankings.

        Parameters
        ----------
        ams1 : ru.AdjacencyMatrix
            The first set of adjacency matrices, represented as a tensor of shape (n, na, na),
            where n is the number of matrices, and na is the number of alternatives.
        ams2 : ru.AdjacencyMatrix
            The second set of adjacency matrices, with the same shape as ams1.
        nu : float or "auto", optional
            The scaling parameter for the kernel. If "auto", it is set to 2 / (na*(na-1)),
            where na is the number of alternatives. The default is "auto".

        Returns
        -------
        ndarray
            A tensor of shape (n, n) representing the Gram matrix of the  Mallows kernel between
            the two sets of adjacency matrices.

        Raises
        ------
        ValueError
            If the two tensors do not have the same shape.

        See Also
        --------
        ru.AdjacencyMatrix : Class representing a ranking.

        Notes
        -----
        The Mallows kernel is defined as:

        .. math::
            K(A_1, A_2) = exp(-nu/2 * \sum_{i < j} |A_{1, i, j} - A_{2, i, j}|)

        where :math:`A_1` and :math:`A_2` are adjacency matrices, and :math:`nu` is a scaling parameter.

        Examples
        --------
        >>> from . import rankings_utils as ru
        >>> ams1 = ru.AdjacencyMatrix(np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]))
        >>> ams2 = ru.AdjacencyMatrix(np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]]))
        >>> MallowsKernel(nu="auto").gram_matrix(ams1, ams2)
        array([[0.60653066, 0.36787944],
               [0.36787944, 0.60653066]])
        """
        self._validate_vectorized_inputs_ams(ams1, ams2)
        self._instantiate_parameters(na=ams1.shape[1])

        ndisc = np.logical_xor(np.expand_dims(ams1, axis=1), np.expand_dims(ams2, axis=0)).sum(axis=(-1, -2))
        return np.exp(-self.nu / 2 * ndisc)