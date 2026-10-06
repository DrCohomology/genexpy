"""
Kernels for experimental results.

- ``rankings``: kernels for rankings (Borda, Jaccard, Mallows), for results expressed as rankings of alternatives.
- ``vectors``: kernels for numerical vectors (RBF), for results expressed as raw scores.
- ``base``: the common base class and the approximation of the quantile function of the MMD.
"""
from . import base, rankings, utils, vectors
from .base import Kernel
from .rankings import RankingKernel, BordaKernel, JaccardKernel, MallowsKernel
from .vectors import VectorKernel, RBFKernel

__all__ = ["base", "rankings", "utils", "vectors",
           "Kernel", "RankingKernel", "BordaKernel", "JaccardKernel", "MallowsKernel", "VectorKernel", "RBFKernel"]
