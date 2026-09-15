"""
Pin down the vectorized-vs-embedding discrepancy in MallowsKernel.mmd_distribution.

Run against YOUR local genexpy (not the pushed one). Three checks, in order of
how likely they are to be the culprit.
"""

import numpy as np

from genexpy.utils import rankings as ru
from genexpy.kernels.rankings import MallowsKernel

# ----------------------------------------------------------------------------
# replace this block with your own sample if you prefer
rng = np.random.default_rng(3)
na, N = 5, 60
rv = np.array([rng.integers(0, 3, size=na) for _ in range(N)]).T  # (na, N), ties allowed
sample = ru.SampleAM.from_rank_vector_matrix(rv)
n, rep, seed = 20, 5000, 11
# ----------------------------------------------------------------------------

support, pmf = sample.get_support_pmf()


# === CHECK 1: does gram_matrix actually use its second argument? ============
# The embedding path only ever calls gram_matrix(x, x); the vectorized path is
# the only one that needs a genuine cross Gram matrix Kxy. So a gram_matrix
# that silently drops its second argument breaks vectorized only.

kernel = MallowsKernel(nu="auto", na=na)
kernel.set_support(support)

ms1, ms2 = sample.get_multisample_pair(subsample_size=n, rep=4, seed=seed,
                                       disjoint=True, replace=False)
x1 = kernel._convert_multisample_to_vectorized_input_format(ru.MultiSampleAM(ms1))
x2 = kernel._convert_multisample_to_vectorized_input_format(ru.MultiSampleAM(ms2))

Kxy = kernel.gram_matrix(x1, x2)
Kxx = kernel.gram_matrix(x1, x1)

ignores_second_arg = np.allclose(Kxy, Kxx)
print("CHECK 1  gram_matrix ignores its second argument:", ignores_second_arg)
print("         diagonal of Kxy all ones (same symptom):",
      np.allclose(np.diagonal(Kxy, axis1=-2, axis2=-1), 1.0))


# === CHECK 2: is nu the same in both paths, and is the cached K consistent? ==
# The embedding path caches self.K and only set_support clears it. The
# vectorized path re-evaluates the kernel on every call. Any nu that is
# data-dependent, or that is assigned after the first embedding call, is
# therefore applied inconsistently between the two.

kernel = MallowsKernel(nu="auto", na=na)
kernel.set_support(support)

nu_before = kernel.nu
_ = kernel.mmd_distribution(sample, n=n, rep=10, seed=seed, method="embedding",
                            use_cached_support_matrix=True)
K_cached = kernel.K.copy()
nu_after = kernel.nu
_ = kernel.mmd_distribution(sample, n=n, rep=10, seed=seed, method="vectorized")

x = kernel._convert_sample_to_input_format(support)
K_fresh = kernel.gram_matrix(x, x)

print("CHECK 2  nu stable across calls:", nu_before == nu_after == kernel.nu,
      f"({nu_before} -> {kernel.nu})")
print("         cached K == K recomputed with the current nu:",
      np.allclose(K_cached, K_fresh))


# === CHECK 3: which method satisfies E[n * MMD_n^2] = 2 * Lambda_1? =========
# Exact for iid draws from the empirical distribution, i.e. disjoint=False and
# replace=True. Under the other three sampling schemes the identity picks up a
# finite-population correction of a few percent, so do not use them here.

kernel = MallowsKernel(nu="auto", na=na)
kernel.set_support(support)
K = kernel.gram_matrix(support, support)
Lambda1 = float(pmf @ np.diag(K) - pmf @ K @ pmf)

print(f"CHECK 3  2 * Lambda_1 = {2 * Lambda1:.6f}")
for method in ["vectorized", "embedding"]:
    kernel.K = None
    mmd = kernel.mmd_distribution(sample, n=n, rep=rep, seed=seed,
                                  disjoint=False, replace=True, method=method)
    print(f"         {method:<11s} n*E[MMD^2] = {n * np.mean(mmd ** 2):.6f}"
          f"   mean MMD = {mmd.mean():.6f}")