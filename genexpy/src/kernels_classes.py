import numpy as np

from numba import njit
from typing import Literal, TypeAlias, Union

from . import rankings_utils as ru


RankVector: TypeAlias = np.ndarray[int]
RankByte: TypeAlias = bytes
Ranking: TypeAlias = Union[RankVector, RankByte]



class Kernel:

    vectorized_input_format: Literal["adjmat", "vector"] = None

    def __init__(self, **kwargs) -> None:
        pass


    def get_eps(self, delta):
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

    @staticmethod
    def _validate_vectorized_inputs_ams(ams1: ru.AdjacencyMatrix, ams2: ru.AdjacencyMatrix):
        if ams1.shape != ams2.shape:
            raise ValueError("The two adjacency matrices' dimensions do not match.")
        if ams1.shape[1] != ams1.shape[2]:
            raise ValueError(f"Input with shape {ams1.shape} is not an array of adjacency matrices.")


    def _instantiate_parameters(self, *args, **kwargs):
        pass


    def __call__(self, x1: Ranking, x2: Ranking, **kwargs) -> float:
        pass


    def _bytes(self, b1: RankByte, b2: RankByte) -> float:
        pass


    def _rv(self, r1: RankVector, r2: RankVector) -> float:
        pass


    def gram_matrix_scalar(self, *args):
        pass


    def gram_matrix(self, *args):
        pass

    def __str__(self):
        return "Kernel"


class BordaKernel(Kernel):

    vectorized_input_format = "vector"

    def __init__(self, idx: int, nu: Union[float, Literal["auto"]] = "auto", **kwargs) -> None:
        super().__init__(**kwargs)
        self.idx = idx
        self.nu = nu
        self._validate_parameters()

        self.gram_matrix = np.vectorize(self.gram_matrix_scalar,
                                        signature="(na, n), (na, n) -> (n, n)", otypes=[float], excluded="self")

    # TODO: Not working if nu has been instantiated with _instantiate_parameters. easy fix is generalize the value of eps
    def get_eps(self, delta):
        if self.nu != "auto":
            raise NotImplementedError(f"Kernel.get_eps not implemented for nu={self.nu}")
        return np.sqrt(2 * (1 - np.exp(-delta)))


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

    # TODO: let the function accept the name of an alternative instead of just indices
    def __call__(self, x1: Ranking, x2: Ranking, use_rv: bool = True) -> float:
        """
        Computes a kernel based on the Borda count for a specific alternative indexed by 'idx'. This kernel considers the
        rescaled difference of the Borda counts at a particular position and is adjusted by a kernel bandwidth 'nu'.

        Parameters:
        - x1 (Ranking): The first ranking, either as a RankVector or RankByte.
        - x2 (Ranking): The second ranking, either as a RankVector or RankByte.
        - idx (int): Index of the alternative under consideration within the ranking.
        - nu (float, 'auto'): The kernel bandwidth, adjusted automatically to the inverse of the number of alternatives squared if 'auto'.
        - use_rv (bool): If True, uses rank vector representation; otherwise expects a byte representation.

        Returns:
        - float: The computed kernel value.

        Raises:
        - ValueError: If the rankings do not have the same number of alternatives.
        """
        self._validate_inputs(x1, x2)
        self._instantiate_parameters(na=len(x1) if use_rv else np.sqrt(len(x1)))

        return self._rv(x1, x2) if use_rv else self._bytes(x1, x2)

    def gram_matrix_scalar(self, rv1: RankVector, rv2: RankVector):
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

        self.gram_matrix = np.vectorize(self.gram_matrix_scalar,
                                        signature="(na, n), (na, n) -> (n, n)", otypes=[float], excluded="self")


    def get_eps(self, delta):
        return np.sqrt(2 * (1 - (1-delta)))

    def _validate_parameters(self):
        if isinstance(self.k, int):
            pass
        else:
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

    def __call__(self, x1: Ranking, x2: Ranking, use_rv=True, **kwargs) -> float:
        """
        Computes the Jaccard kernel between two rankings by considering the top k tiers of the rankings. This kernel
        measures the similarity based on the intersection over union of the rankings within the top k tiers.

        Parameters:
        - x1 (Ranking): The first ranking as a RankVector or RankByte.
        - x2 (Ranking): The second ranking as a RankVector or RankByte.
        - k (int): The number of top tiers to consider for the similarity calculation.
        - use_rv (bool): Determines whether to use the rank vector or byte representation for the calculation.

        Returns:
        - float: The computed Jaccard similarity score.

        Raises:
        - ValueError: If the rankings do not have the same number of alternatives.
        """
        self._validate_inputs(x1, x2)
        return self._rv(x1, x2) if use_rv else self._bytes(x1, x2)


    def gram_matrix_scalar(self, rv1: RankVector, rv2: RankVector):
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

        Examples
        --------
        >>> from . import ranking_utils as ru
        >>> rv1 = ru.RankVector(np.array([[0, 1, 2], [2, 0, 1]]))
        >>> rv2 = ru.RankVector(np.array([[1, 0, 2], [0, 2, 1]]))
        >>> JaccardKernel(k=2).gram_matrix(rv1, rv2)
        array([[0.66666667, 0.5       ],
               [0.5       , 0.66666667]])
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

        self.gram_matrix = np.vectorize(self.gram_matrix_scalar,
                                        signature="(n, na, na), (n, na, na) -> (n, n)", otypes=[float], excluded="self")

    def get_eps(self, delta):
        if self.nu != "auto":
            raise NotImplementedError(f"MallowsKernel.get_eps not implemented for nu={self.nu}")
        return np.sqrt(2 * (1 - np.exp(-delta)))

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

    # @np.vectorize(signature="(n, na, na), (n, na, na) -> (n, n)", otypes=[float])
    def gram_matrix_scalar(self, ams1: ru.AdjacencyMatrix, ams2: ru.AdjacencyMatrix):
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

