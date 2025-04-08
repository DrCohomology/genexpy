"""
Kernels applies to different rankings..
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns

from collections import Counter, defaultdict
from importlib import reload
from itertools import product
from numba import njit
from pathlib import Path
from scipy.special import binom
from scipy.stats import binomtest, friedmanchisquare, spearmanr, wilcoxon
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import time
from tqdm.auto import tqdm
from typing import Union, TypeAlias, Literal
from genexpy import kernels as ku
from genexpy import rankings_utils as ru
from genexpy import mmd as mmd
from genexpy import probability_distributions as du

from genexpy import kernels_vectorized as kvu



# r1 = ru.AdjacencyMatrix.from_rank_vector(np.array([0, 0, 1]))
# r2 = ru.AdjacencyMatrix.from_rank_vector(np.array([1, 1, 0]))

r1 = np.array([0, 0, 0])
r2 = np.array([0, 1, 1])

mallows = ku.mallows_kernel(r1, r2, use_rv=True)
jaccard = ku.jaccard_kernel(r1, r2)
borda = ku.borda_kernel(r1, r2, idx=0)

print(mallows, jaccard, borda)
