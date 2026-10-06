"""
Kernels for rankings.

A ranking of na alternatives is stored either as a rank vector (``r[i]`` is the rank of alternative i,
0 = best, ties allowed) or as the bytes of its adjacency matrix (see ``genexpy.utils.rankings``).
Every kernel is normalized, k(r, r) = 1, and takes values in [0, 1].

- ``BordaKernel(alternative=..., nu=...)``: are the results similar w.r.t. the position of one alternative?
- ``JaccardKernel(t=...)``: are the results similar w.r.t. the alternatives in the top-t tiers?
- ``MallowsKernel(nu=...)``: are the results similar w.r.t. the whole ranking?

Besides the kernel itself, every kernel estimates the distribution of the MMD between two samples of n
rankings drawn from a sample of rankings (``mmd_distribution``), which is what the external validity
analysis is built on.
"""

import numbers
import warnings

import numpy as np
import pandas as pd

from typing import Literal, TypeAlias, Union

from genexpy.utils import rankings as ru
from genexpy.kernels import base

RankVector: TypeAlias = np.ndarray[int]
RankByte: TypeAlias = bytes
Ranking: TypeAlias = Union[RankVector, RankByte]

MMDMethod: TypeAlias = Literal["auto", "naive", "vectorized", "embedding", "approximation"]


def _validate_bandwidth(name: str, value) -> float:
    """Return `value` as a float if it is a valid (non-negative, real) kernel bandwidth, raise otherwise."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or value < 0:
        raise ValueError(f"Invalid value for parameter {name}={value}. Accepted: non-negative float or 'auto'.")
    return float(value)


class RankingKernel(base.Kernel):
    """
    Base class of the kernels for rankings.

    Parameters
    ----------
    support : ru.UniverseAM, optional
        The rankings over which the Gram matrix used by the 'embedding' method is computed and cached.
        Set it with ``set_support``. If None, the support is recomputed at every call.
    """
    vectorized_input_format: Literal["adjmat", "vector"] = None

    def __init__(self, support: ru.UniverseAM = None, **kwargs) -> None:
        super().__init__()
        self.support = support  # support of rankings
        self.K = None  # gram matrix of the support

    def set_support(self, support: ru.UniverseAM):
        """Set the rankings on which the Gram matrix is cached, and clear the cache."""
        self.support = support
        self.K = None

    def get_eps(self, delta: float, na: int = None) -> float:
        """MMD threshold epsilon corresponding to the kernel-specific similarity threshold delta."""
        raise NotImplementedError

    def _resolve_na(self, na: int = None) -> int:
        na = na if na is not None else getattr(self, "na", None)
        if na is None:
            raise ValueError("The number of alternatives na must be passed.")
        return na

    def _validate_parameters(self):
        pass

    @staticmethod
    def _validate_inputs(x1: Ranking, x2: Ranking):
        if len(x1) != len(x2):
            raise ValueError("Ranking dimensions do not match")
        if isinstance(x1, RankByte):
            if np.sqrt(len(x1)) != int(np.sqrt(len(x1))):
                raise ValueError(f"The input bytestring has length {len(x1)} and is not a square (adjacency) matrix.")

    @staticmethod
    def _validate_vectorized_inputs_rv(rv1: RankVector, rv2: RankVector):
        """Rank matrices have shape (..., na, n): the number of alternatives na must coincide."""
        for rv in (rv1, rv2):
            if np.ndim(rv) < 2:
                raise ValueError(
                    f"The shape of the input should be that of a valid rank matrix, i.e., (num_alternatives, "
                    f"num_rankings). Object of shape {np.shape(rv)} is not a valid rank matrix.")
        if rv1.shape[-2] != rv2.shape[-2]:
            raise ValueError(f"The two rank matrices' dimensions do not match: {rv1.shape} and {rv2.shape}.")

    @staticmethod
    def _validate_vectorized_inputs_ams(ams1: ru.AdjacencyMatrix, ams2: ru.AdjacencyMatrix):
        """Arrays of adjacency matrices have shape (..., n, na, na): na must coincide."""
        for ams in (ams1, ams2):
            if np.ndim(ams) < 3:
                raise ValueError(
                    f"The shape of the input should be that of an array of adjacency matrices, i.e., (num_rankings, "
                    f"num_alternatives, num_alternatives). Object of shape {np.shape(ams)} is not valid.")
            if ams.shape[-1] != ams.shape[-2]:
                raise ValueError(f"Input with shape {ams.shape} is not an array of adjacency matrices (the last two "
                                 f"dimensions should coincide).")
        if ams1.shape[-1] != ams2.shape[-1]:
            raise ValueError(f"The two arrays of adjacency matrices' dimensions do not match: "
                             f"{ams1.shape} and {ams2.shape}.")

    def _set_parameters(self, *args, **kwargs):
        pass

    def __call__(self, x1: Ranking, x2: Ranking, use_rv: bool = True) -> float:
        """
        Kernel between two rankings.

        Parameters
        ----------
        x1, x2 : RankVector or RankByte
            The two rankings, as rank vectors (``use_rv=True``) or as bytes of adjacency matrices (``use_rv=False``).
        use_rv : bool
            Whether the inputs are rank vectors.

        Returns
        -------
        float
            The kernel, in [0, 1].
        """
        self._validate_inputs(x1, x2)
        return self._rv(x1, x2) if use_rv else self._bytes(x1, x2)

    def _bytes(self, b1: RankByte, b2: RankByte) -> float:
        raise NotImplementedError

    def _rv(self, r1: RankVector, r2: RankVector) -> float:
        raise NotImplementedError

    def _gram_matrix_naive(self, s1: ru.SampleAM, s2: ru.SampleAM, use_rv: bool = True) -> np.ndarray[float]:
        """
        Gram matrix between two samples of rankings, evaluating the kernel on every pair (reference implementation).

        Parameters
        ----------
        s1, s2 : ru.SampleAM
            The two samples of rankings.
        use_rv : bool
            If True, the rankings are converted to rank vectors before evaluating the kernel.

        Returns
        -------
        np.ndarray
            out[i, j] = k(s1[i], s2[j]).
        """
        if use_rv:
            s1 = s1.to_rank_vector_matrix().T  # rows: rankings, cols: alternatives
            s2 = s2.to_rank_vector_matrix().T

        out = np.zeros((len(s1), len(s2)))
        if len(s1) == len(s2) and np.array_equal(s1, s2):
            for i2, x2 in enumerate(s2):
                for i1 in range(i2):
                    out[i1, i2] = self(s1[i1], x2, use_rv)
            d = np.diag([self(x, x, use_rv) for x in s1])
            return out + d + out.T
        for i1, x1 in enumerate(s1):
            for i2, x2 in enumerate(s2):
                out[i1, i2] = self(x1, x2, use_rv)
        return out

    def _gram_matrix_scalar(self, x1, x2) -> np.ndarray[float]:
        """
        Gram matrix between two sets of rankings in the kernel's input format (see ``gram_matrix``).
        Leading (batch) dimensions are broadcast.
        """
        raise NotImplementedError()

    def _gram_matrix_vectorized(self, x1, x2) -> np.ndarray[float]:
        return self._gram_matrix_scalar(x1, x2)

    def gram_matrix(self, sample1, sample2=None) -> np.ndarray[float]:
        """
        Gram matrix between two samples of rankings, K[i, j] = k(sample1[i], sample2[j]).

        Parameters
        ----------
        sample1 : ru.SampleAM or np.ndarray
            A sample of rankings, or an array already in the kernel's input format: a rank matrix of shape
            (na, n) for ``BordaKernel`` and ``JaccardKernel``, an array of adjacency matrices of shape (n, na, na)
            for ``MallowsKernel``. Arrays can have leading batch dimensions, in which case one Gram matrix is
            computed per batch.
        sample2 : same as sample1, optional
            The second sample. If None, sample1 is used.

        Returns
        -------
        np.ndarray
            The Gram matrix, of shape (..., n1, n2).
        """
        x1 = self._convert_sample_to_input_format(sample1)
        x2 = x1 if sample2 is None else self._convert_sample_to_input_format(sample2)
        try:
            return self._gram_matrix_vectorized(x1, x2)
        except NotImplementedError:
            return self._gram_matrix_naive(sample1, sample1 if sample2 is None else sample2)

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

    def _convert_sample_to_input_format(self, s: ru.UniverseAM):

        if not isinstance(s, ru.UniverseAM):
            return s
        if not isinstance(s, ru.SampleAM):
            s = s.view(ru.SampleAM)

        na = int(np.sqrt(len(s[0])))

        # Input checks
        if np.not_equal(na, np.sqrt(len(s[0]))):
            raise ValueError(f"The first SampleAM has invalid dimension: the length of s[0] is "
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
        """MMD of every pair of subsamples, evaluating the kernel on every pair of rankings (slow, reference)."""
        ms1, ms2 = sample.get_multisample_pair(subsample_size=n, rep=rep, seed=seed, disjoint=disjoint, replace=replace)

        mmd = []
        for s1, s2 in zip(ms1, ms2):
            s1 = ru.SampleAM(s1)
            s2 = ru.SampleAM(s2)

            Kxx = self._gram_matrix_naive(s1, s1)
            Kyy = self._gram_matrix_naive(s2, s2)
            Kxy = self._gram_matrix_naive(s1, s2)

            mmd.append(np.sqrt(np.abs(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())))

        return np.array(mmd)

    def _mmd_distribution_vectorized(self, sample: ru.SampleAM, n: int, rep: int, seed: int = 0, disjoint: bool = True,
                                     replace: bool = False) -> np.ndarray[float]:
        """MMD of every pair of subsamples, from the three (batched) Gram matrices of every pair."""
        ms1, ms2 = sample.get_multisample_pair(subsample_size=n, rep=rep, seed=seed, disjoint=disjoint, replace=replace)

        ms1 = ru.MultiSampleAM(ms1)
        ms2 = ru.MultiSampleAM(ms2)

        x1 = self._convert_multisample_to_vectorized_input_format(ms1)
        x2 = self._convert_multisample_to_vectorized_input_format(ms2)

        Kxx = self.gram_matrix(x1, x1)
        Kxy = self.gram_matrix(x1, x2)
        Kyy = self.gram_matrix(x2, x2)

        return np.sqrt(np.abs(np.mean(Kxx, axis=(1, 2)) + np.mean(Kyy, axis=(1, 2)) - 2 * np.mean(Kxy, axis=(1, 2))))

    def _mmd_distribution_embedding(self, sample: ru.SampleAM, n: int, rep: int, seed: int = 0,
                                    disjoint: bool = True, replace: bool = False,
                                    use_cached_support_matrix: bool = False) -> np.ndarray[float]:
        """
        MMD of every pair of subsamples, as sqrt(alpha^T K alpha), where alpha is the difference of the empirical
        pmfs of the pair over the support and K is the Gram matrix of the support (computed once).
        """

        if use_cached_support_matrix and self.support is None:
            raise ValueError("To cache the support matrix, self.support must be set.")

        ms1, ms2 = sample.get_multisample_pair(subsample_size=n, rep=rep, seed=seed,
                                               disjoint=disjoint, replace=replace)

        ms1 = ru.MultiSampleAM(ms1)
        ms2 = ru.MultiSampleAM(ms2)

        # ms1[0] is compared to ms2[0] etc..., i.e. alpha[:, 0] pairs ms1[0] with ms2[0]
        alpha, support = ms1.get_alpha(ms2, support=self.support)

        if not (use_cached_support_matrix and self.K is not None):
            x = self._convert_sample_to_input_format(support)
            self.K = self.gram_matrix(x, x)

        if self.K.shape[0] != len(support):
            raise ValueError(f"The cached support matrix is {self.K.shape[0]}x{self.K.shape[0]} but the support "
                             f"holds {len(support)} rankings. Call set_support to clear the cache.")

        # only the diagonal of alpha.T @ K @ alpha is needed, so the rep x rep
        # product is never formed. The absolute value is to avoid machine 0-s.
        return np.sqrt(np.abs(np.einsum("ir,ir->r", alpha, self.K @ alpha)))

    def _mmd_icdf_approximation(self, sample: ru.SampleAM, n: int, rep: int, alpha_min: float = 0.6,
                                alpha_max: float = 1) -> np.ndarray[float]:
        """
        Close-form approximation of the quantile function (ICDF) of the MMD at `rep` equispaced levels in
        [alpha_min, alpha_max). See ``base.approximate_mmd_icdf``.
        """
        support, pmf = sample.get_support_pmf()
        x = self._convert_sample_to_input_format(support)
        K = self.gram_matrix(x, x)
        return base.approximate_mmd_icdf(K, pmf, n=n, rep=rep, alpha_min=alpha_min, alpha_max=alpha_max)

    # TODO implement
    @staticmethod
    def _select_fastest_mmd_estimation_method() -> Literal["naive", "vectorized", "embedding", "approximation"]:
        return "embedding"

    def mmd_distribution(self, sample: ru.SampleAM, n: int, rep: int, seed: int = 0, disjoint: bool = True,
                         replace: bool = False, method: MMDMethod = "auto",
                         use_cached_support_matrix: bool = False, alpha_min=0.7, alpha_max=1) -> np.ndarray[float]:
        """
        Estimate the distribution of the MMD between two samples of n rankings drawn from `sample`.

        Parameters
        ----------
        sample : ru.SampleAM
            The rankings to draw the pairs of subsamples from.
        n : int
            Size of each subsample.
        rep : int
            Number of pairs of subsamples, i.e., size of the output.
        seed : int
            Random seed for the subsampling.
        disjoint : bool
            If True, the two subsamples are drawn from two disjoint halves of `sample`.
        replace : bool
            If True, the subsamples are drawn with replacement.
        method : {"auto", "naive", "vectorized", "embedding", "approximation"}
            - "embedding" (default with "auto", fast): MMD from the difference of the empirical pmfs.
            - "vectorized": MMD from the Gram matrices of every pair of subsamples.
            - "naive" (slow): as "vectorized", evaluating the kernel on every pair of rankings.
            - "approximation": close-form approximation; the output is NOT a sample of the MMD, but its
              quantile function evaluated at `rep` equispaced levels in [alpha_min, alpha_max).
        use_cached_support_matrix : bool
            For method="embedding": reuse the Gram matrix of ``self.support`` across calls.
        alpha_min, alpha_max : float
            For method="approximation": the range of quantile levels.

        Returns
        -------
        np.ndarray
            Array of shape (rep, ).
        """

        if method == "auto":
            method = self._select_fastest_mmd_estimation_method()

        match method:
            case "naive":
                return self._mmd_distribution_naive(sample, n=n, rep=rep, seed=seed,
                                                    disjoint=disjoint, replace=replace)
            case "vectorized":
                return self._mmd_distribution_vectorized(sample, n=n, rep=rep, seed=seed,
                                                         disjoint=disjoint, replace=replace)
            case "embedding":
                return self._mmd_distribution_embedding(sample, n=n, rep=rep, seed=seed,
                                                        disjoint=disjoint, replace=replace,
                                                        use_cached_support_matrix=use_cached_support_matrix)
            case "approximation":
                warnings.warn("The output of calling the function with method=approximation is not a sample of the MMD"
                              " but its icdf.")
                return self._mmd_icdf_approximation(sample, n=n, rep=rep, alpha_min=alpha_min, alpha_max=alpha_max)
            case _:
                raise ValueError(f"Invalid method {method}.")

    def mmd_distribution_many_n(self, sample: ru.SampleAM, nmin: int, nmax: int, step: int,
                                seed: int = 100, disjoint: bool = True, replace: bool = False, N: int = None,
                                method: MMDMethod = "auto", **mmd_distribution_parms) -> pd.DataFrame:
        """
        Run ``mmd_distribution`` for n in range(nmin, nmax, step) (seed * n is the seed for size n).

        Parameters
        ----------
        sample, seed, disjoint, replace, method :
            See ``mmd_distribution``.
        nmin, nmax, step : int
            The subsample sizes n, as in ``range(nmin, nmax, step)``.
        N : int, optional
            Size of `sample`, only stored in the output.
        **mmd_distribution_parms :
            Passed to ``mmd_distribution`` (e.g., rep, use_cached_support_matrix).

        Returns
        -------
        pd.DataFrame
            Columns: n, mmd, method, N, disjoint, replace, kernel.
        """
        mmds = {
            n: self.mmd_distribution(sample=sample, n=n, seed=seed * n, disjoint=disjoint, replace=replace,
                                     method=method, **mmd_distribution_parms)
            for n in range(nmin, nmax+1, step)}

        dfmmd = pd.DataFrame(mmds).melt(var_name="n", value_name="mmd")
        dfmmd["method"] = method
        dfmmd["N"] = N
        dfmmd["disjoint"] = disjoint
        dfmmd["replace"] = replace
        dfmmd["kernel"] = self.__str__()

        return dfmmd


class BordaKernel(RankingKernel):
    """
    Borda kernel: k(r1, r2) = exp(-nu * |b1 - b2|), where b is the number of alternatives ranked no better than the
    alternative of interest (its Borda count).

    Goal: the results should agree on the position of one alternative.

    Parameters
    ----------
    idx : int, optional
        Index (row) of the alternative of interest. Exactly one of idx and alternative must be passed.
    alternative : str, optional
        Name of the alternative of interest; requires ordered_alternatives.
    nu : float or "auto"
        Bandwidth. "auto" sets nu = 1 / (na - 1) and requires na.
    na : int, optional
        Number of alternatives.
    ordered_alternatives : array-like, optional
        Names of the alternatives, in the order of the rows of the rank matrices.

    Examples
    --------
    >>> k = BordaKernel(idx=0, na=4)
    >>> k(np.array([0, 1, 2, 3]), np.array([3, 2, 1, 0]))   # exp(-1)
    """
    vectorized_input_format = "vector"

    def __init__(self, idx: int = None, alternative: str = None, nu: Union[float, Literal["auto"]] = "auto",
                 na: int = None, ordered_alternatives: np.ndarray = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if alternative is None and idx is None:
            raise ValueError("Exactly one of alternative and idx must be specified.")
        elif alternative is not None and idx is not None:
            raise ValueError("Exactly one of alternative and idx must be specified.")
        self.alternative = alternative
        self.idx = idx
        self.na = na

        self.nu = nu
        self._set_parameters(na=na, ordered_alternatives=ordered_alternatives)
        self._validate_parameters()

    def __repr__(self):
        return f"BordaKernel(nu={self.nu:.5f}, idx={self.idx})"

    def latex_str(self):
        return fr"$k_\text{{b}}^{{\nu={self.nu:.3f}, a^*={self.idx}}}$"

    def get_eps(self, delta: float, na: int = None) -> float:
        """
        epsilon(delta) = sqrt(2 * (1 - exp(-nu * (na - 1) * delta))), where delta is the maximum difference between
        the fractions of alternatives ranked no better than the alternative of interest (|b1 - b2| / (na - 1)).
        """
        na = self._resolve_na(na)
        return np.sqrt(2 * (1 - np.exp(-self.nu * (na - 1) * delta)))

    def _validate_parameters(self):
        self.nu = _validate_bandwidth("nu", self.nu)

        if isinstance(self.idx, bool) or not isinstance(self.idx, numbers.Integral):
            raise ValueError(f"Invalid value for parameter idx={self.idx}. Accepted: int")
        self.idx = int(self.idx)
        if self.na is not None and not 0 <= self.idx < self.na:
            raise ValueError(f"Parameter idx={self.idx} is out of range for na={self.na} alternatives.")

    def _validate_inputs(self, x1: Ranking, x2: Ranking):
        if len(x1) != len(x2):
            raise ValueError(f"The rankings have different lengths {len(x1)} and {len(x2)}")
        na = len(x1)
        if isinstance(x1, RankByte):
            na = int(np.sqrt(len(x1)))
            if na ** 2 != len(x1):
                raise ValueError(f"The input bytestring has length {len(x1)} and is not a square (adjacency) matrix.")
        if self.idx >= na:
            raise ValueError(f"The idx must not exceed the number of alternatives.")

    def _set_parameters(self, na: int = None, ordered_alternatives: np.array = None):
        if self.nu == "auto":
            if na is None or na <= 1:
                raise ValueError("If nu == 'auto', parameter na >= 2 has to be passed.")
            self.nu = 1 / (na - 1)

        if self.idx is None and self.alternative is not None:
            if ordered_alternatives is None:
                raise ValueError("If idx is None, parameter ordered_alternatives has to be passed.")
            ordered_alternatives = list(np.asarray(ordered_alternatives))
            if self.alternative not in ordered_alternatives:
                raise ValueError(f"Alternative {self.alternative} is not among the alternatives.")
            self.idx = ordered_alternatives.index(self.alternative)

    def _rv(self, r1: RankVector, r2: RankVector) -> float:
        return np.exp(- self.nu * np.abs(np.sum(r1 >= r1[self.idx]) - np.sum(r2 >= r2[self.idx])))

    def _bytes(self, b1: RankByte, b2: RankByte) -> float:
        # row idx of the adjacency matrix: A[idx, j] = r[idx] <= r[j], i.e., the Borda count of idx
        na = int(np.sqrt(len(b1)))
        d1 = np.frombuffer(b1, dtype=np.int8).reshape(na, na)[self.idx].astype(int).sum()
        d2 = np.frombuffer(b2, dtype=np.int8).reshape(na, na)[self.idx].astype(int).sum()
        return np.exp(- self.nu * np.abs(d1 - d2))

    def _gram_matrix_scalar(self, rv1: RankVector, rv2: RankVector):
        """
        Gram matrix of the Borda kernel between two sets of rankings, represented as rank matrices.

        Parameters
        ----------
        rv1 : RankVector
            Rank matrix of shape (..., na, n): column j is the rank vector of ranking j.
        rv2 : RankVector
            Rank matrix of shape (..., na, m).

        Returns
        -------
        np.ndarray
            Array of shape (..., n, m), K[..., i, j] = exp(-nu * |d1[i] - d2[j]|), where d is the number of
            alternatives ranked no better than alternative idx.
        """
        self._validate_vectorized_inputs_rv(rv1, rv2)

        d1 = np.sum(rv1 >= rv1[..., [self.idx], :], axis=-2)  # dominated, (..., n)
        d2 = np.sum(rv2 >= rv2[..., [self.idx], :], axis=-2)  # (..., m)
        return np.exp(- self.nu * np.abs(d1[..., :, None] - d2[..., None, :]))


class JaccardKernel(RankingKernel):
    """
    Jaccard kernel: k(r1, r2) = |T1 & T2| / |T1 | T2|, where T is the set of alternatives in the top-t tiers.

    Goal: the results should agree on the best alternatives.

    Parameters
    ----------
    t : int
        Number of top tiers considered (t=1: only the best alternatives, including ties).
    """
    vectorized_input_format = "vector"

    def __init__(self, t: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.t = t
        self._validate_parameters()

    def __repr__(self):
        return f"JaccardKernel(t={self.t})"

    def get_eps(self, delta: float, na: int = None) -> float:
        """epsilon(delta) = sqrt(2 * delta), where delta is the maximum Jaccard distance 1 - k."""
        return np.sqrt(2 * (1 - (1 - delta)))

    def _validate_parameters(self):
        if isinstance(self.t, bool) or not isinstance(self.t, numbers.Integral) or self.t < 1:
            raise ValueError(f"Invalid value for parameter t={self.t}. Accepted: positive int")
        self.t = int(self.t)

    def _bytes(self, b1: RankByte, b2: RankByte) -> float:
        na = int(np.sqrt(len(b1)))
        r1 = ru.AdjacencyMatrix.from_bytes(b1, (na, na)).to_rank_vector()
        r2 = ru.AdjacencyMatrix.from_bytes(b2, (na, na)).to_rank_vector()
        return self._rv(r1, r2)

    def _rv(self, r1: RankVector, r2: RankVector) -> float:
        """
        Supports tied rankings as columns of the output from SampleAM.to_rank_vector_matrix().
        """
        topk1 = np.where(r1 < self.t)[0]
        topk2 = np.where(r2 < self.t)[0]

        return len(set(topk1).intersection(set(topk2))) / len(set(topk1).union(set(topk2)))

    def _gram_matrix_scalar(self, rv1: RankVector, rv2: RankVector):
        r"""
        Gram matrix of the Jaccard kernel between two sets of rankings, represented as rank matrices.

        Parameters
        ----------
        rv1 : RankVector
            Rank matrix of shape (..., na, n): column j is the rank vector of ranking j (dense ranks, 0 = best).
        rv2 : RankVector
            Rank matrix of shape (..., na, m).

        Returns
        -------
        np.ndarray
            Array of shape (..., n, m), K[..., i, j] = |T_i \cap T_j| / |T_i \cup T_j|, with T the set of
            alternatives with rank < t.
        """
        self._validate_vectorized_inputs_rv(rv1, rv2)

        k1 = (rv1 < self.t).astype(float)  # (..., na, n)
        k2 = (rv2 < self.t).astype(float)  # (..., na, m)
        intersection = np.swapaxes(k1, -1, -2) @ k2  # (..., n, m), exact integer counts
        union = k1.sum(axis=-2)[..., :, None] + k2.sum(axis=-2)[..., None, :] - intersection
        return intersection / union

    def latex_str(self):
        return fr"$k_\text{{j}}^{{t={self.t}}}$"


class MallowsKernel(RankingKernel):
    """
    Mallows kernel: k(r1, r2) = exp(-nu * n_d(r1, r2)), where n_d is the number of discordant pairs of alternatives
    (a pair tied in one ranking only counts 1/2).

    Goal: the results should agree on the whole ranking.

    Parameters
    ----------
    nu : float or "auto"
        Bandwidth. "auto" sets nu = 1 / binom(na, 2) and requires na.
    na : int, optional
        Number of alternatives.
    """
    vectorized_input_format = "adjmat"

    def __init__(self, nu: Union[float, Literal["auto"]] = "auto", na: int = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.na = na
        self.nu = nu
        self._set_parameters(na)
        self._validate_parameters()

    def __repr__(self):
        return f"MallowsKernel(nu={self.nu:.5f})"

    def get_eps(self, delta: float, na: int = None) -> float:
        """
        epsilon(delta) = sqrt(2 * (1 - exp(-nu * binom(na, 2) * delta))), where delta is the maximum fraction of
        discordant pairs.
        """
        na = self._resolve_na(na)
        return np.sqrt(2 * (1 - np.exp(- self.nu * (na * (na - 1)) / 2 * delta)))

    def _validate_parameters(self):
        self.nu = _validate_bandwidth("nu", self.nu)

    def _set_parameters(self, na):
        if self.nu == "auto":
            if na is None or na <= 1:
                raise ValueError("If nu == 'auto', parameter na >= 2 has to be passed.")
            self.nu = 2 / (na * (na - 1))

    def _bytes(self, b1: RankByte, b2: RankByte) -> float:
        i1 = np.frombuffer(b1, dtype=np.int8)
        i2 = np.frombuffer(b2, dtype=np.int8)
        return np.exp(- self.nu * np.sum(np.abs(i1 - i2)) / 2)

    def _rv(self, r1: RankVector, r2: RankVector) -> float:
        # twice the number of discordant pairs ((tie, not-tie) counts as 1/2 discordant)
        r1 = np.asarray(r1, dtype=np.int64)
        r2 = np.asarray(r2, dtype=np.int64)
        s1 = np.sign(r1[:, None] - r1[None, :])
        s2 = np.sign(r2[:, None] - r2[None, :])
        out = np.abs(s1 - s2).sum() / 2  # every unordered pair appears twice
        return np.exp(- self.nu * out / 2)

    def _gram_matrix_scalar(self, ams1: ru.AdjacencyMatrix, ams2: ru.AdjacencyMatrix):
        r"""
        Gram matrix of the Mallows kernel between two sets of rankings, represented as adjacency matrices.

        Parameters
        ----------
        ams1 : np.ndarray
            Adjacency matrices, of shape (..., n, na, na).
        ams2 : np.ndarray
            Adjacency matrices, of shape (..., m, na, na).

        Returns
        -------
        np.ndarray
            Array of shape (..., n, m), K[..., i, j] = exp(-nu/2 * \sum_{a, b} |A_i[a, b] - A_j[a, b]|).

        Notes
        -----
        For binary vectors, the number of differing entries is |a| + |b| - 2 a.b, so the counts come from a
        single matrix product instead of an (n, m, na, na) tensor.
        """
        self._validate_vectorized_inputs_ams(ams1, ams2)

        na = ams1.shape[-1]
        a1 = (np.asarray(ams1) != 0).reshape(*ams1.shape[:-2], na * na).astype(float)  # (..., n, na^2)
        a2 = (np.asarray(ams2) != 0).reshape(*ams2.shape[:-2], na * na).astype(float)  # (..., m, na^2)
        ndisc = (a1.sum(axis=-1)[..., :, None] + a2.sum(axis=-1)[..., None, :]
                 - 2 * (a1 @ np.swapaxes(a2, -1, -2)))  # exact integer counts
        return np.exp(-self.nu / 2 * ndisc)

    def latex_str(self):
        return fr"$k_\text{{m}}^{{\nu={self.nu:.3f}}}$"
