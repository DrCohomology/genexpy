"""
genexpy: kernel-based external validity of experimental studies.

Subpackages and modules
-----------------------
- ``genexpy.utils``: rankings as adjacency matrices, samples of rankings, conversion of results into rankings.
- ``genexpy.kernels``: kernels for rankings (Borda, Jaccard, Mallows) and for vectors (RBF), and the estimation
  of the distribution of the MMD.
- ``genexpy.random``: probability distributions over rankings.
- ``genexpy.managers``: ``ProjectManager`` (external validity analysis of a results file, driven by a
  configuration file) and ``PlotManager`` (plots). Import it explicitly: ``from genexpy.managers import ...``.
"""

from . import kernels, utils, random
from .kernels.base import Kernel
from .kernels.rankings import RankingKernel, BordaKernel, JaccardKernel, MallowsKernel
from .kernels.vectors import VectorKernel, RBFKernel
from .utils.rankings import AdjacencyMatrix, UniverseAM, SampleAM, MultiSampleAM, get_matrix_from_df
from .utils.relations import score2rv, vec2rv
from .random import (ProbabilityDistribution, UniformDistribution, DegenerateDistribution, MDegenerateDistribution,
                     SpikeDistribution, PMFDistribution, get_unique_ranks_distribution)

# Legacy aliases, previously exposed by `from .kernels import *` and `from .utils import *`
from .kernels import base, vectors
from .utils import rankings, relations

__all__ = [
    "kernels", "utils", "random",
    "Kernel", "RankingKernel", "BordaKernel", "JaccardKernel", "MallowsKernel", "VectorKernel", "RBFKernel",
    "AdjacencyMatrix", "UniverseAM", "SampleAM", "MultiSampleAM", "get_matrix_from_df", "score2rv", "vec2rv",
    "ProbabilityDistribution", "UniformDistribution", "DegenerateDistribution", "MDegenerateDistribution",
    "SpikeDistribution", "PMFDistribution", "get_unique_ranks_distribution",
]

# Define the genexpy version
__version__ = "0.0.0"
