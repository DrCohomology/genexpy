"""
Locate the vectorized-vs-embedding discrepancy in MallowsKernel.mmd_distribution.

The two methods share the subsampling call and both bottom out in
_gram_matrix_scalar, so they can only diverge in six places. Each stage below
tests exactly one of them and prints OK or FAIL. The first FAIL is the bug.

Run against your local genexpy. Edit only the SAMPLE block.
"""

import numpy as np

from genexpy.utils import rankings as ru
from genexpy.kernels.rankings import MallowsKernel

# --- SAMPLE ------------------------------------------------------------------
rng = np.random.default_rng(3)
na, N = 5, 60
rv = np.array([rng.integers(0, 3, size=na) for _ in range(N)]).T  # (na, N), ties allowed
sample = ru.SampleAM.from_rank_vector_matrix(rv)
n, rep, seed = 6, 200, 11
disjoint, replace = True, False
# -----------------------------------------------------------------------------

support, pmf = sample.get_support_pmf()
kernel = MallowsKernel(nu="auto", na=na)
kernel.set_support(support)

results = []


def report(stage, ok, detail=""):
    results.append((stage, ok))
    print(f"[{'OK  ' if ok else 'FAIL'}] {stage}" + (f"\n       {detail}" if detail else ""))


# === 1. the subsample pairs ==================================================
a1, a2 = sample.get_multisample_pair(subsample_size=n, rep=rep, seed=seed,
                                     disjoint=disjoint, replace=replace)
b1, b2 = sample.get_multisample_pair(subsample_size=n, rep=rep, seed=seed,
                                     disjoint=disjoint, replace=replace)
same_pairs = np.array_equal(np.asarray(a1), np.asarray(b1)) and \
             np.array_equal(np.asarray(a2), np.asarray(b2))
report("1. subsampling is reproducible under a fixed seed", same_pairs,
       "" if same_pairs else "the two methods are not even comparing the same subsamples")

ms1, ms2 = ru.MultiSampleAM(a1), ru.MultiSampleAM(a2)


# === 2. the two conversion routes to adjacency matrices ======================
# vectorized route: MultiSampleAM.to_adjacency_matrices  -> (rep, n, na, na)
# embedding route:  UniverseAM.to_adjmat_array           -> (m, na, na)
x1 = kernel._convert_multisample_to_vectorized_input_format(ms1)
x2 = kernel._convert_multisample_to_vectorized_input_format(ms2)
xs = kernel._convert_sample_to_input_format(support)

by_multisample = {bytes(r): np.asarray(A) for r, A in zip(np.asarray(ms1).ravel(),
                                                          x1.reshape(-1, na, na))}
by_sample = {bytes(r): np.asarray(A) for r, A in zip(np.asarray(support), xs)}
shared = set(by_multisample) & set(by_sample)
conv_ok = bool(shared) and all(np.array_equal(by_multisample[r], by_sample[r]) for r in shared)
report("2. both conversion routes give the same adjacency matrix", conv_ok,
       "" if conv_ok else f"{sum(not np.array_equal(by_multisample[r], by_sample[r]) for r in shared)}"
                          f"/{len(shared)} rankings convert differently")


# === 3. gram_matrix uses its second argument =================================
Kxy = np.asarray(kernel.gram_matrix(x1, x2))
Kxx = np.asarray(kernel.gram_matrix(x1, x1))
uses_second = not np.allclose(Kxy, Kxx)
report("3. gram_matrix does not ignore its second argument", uses_second,
       "" if uses_second else "gram_matrix(x1,x2) == gram_matrix(x1,x1): the cross term Kxy is "
                              "actually Kxx, which breaks vectorized only (embedding only ever "
                              "calls gram_matrix(x,x))")

# entrywise against direct scalar evaluation, on one repetition
A1, A2 = x1[0], x2[0]
K_ref = np.array([[kernel._gram_matrix_scalar(A1[[i]], A2[[j]])[0, 0] for j in range(n)]
                  for i in range(n)])
entry_ok = np.allclose(Kxy[0], K_ref)
report("3b. gram_matrix entries match _gram_matrix_scalar", entry_ok,
       "" if entry_ok else f"max entrywise error {np.abs(Kxy[0] - K_ref).max():.3e} -- "
                           f"suspect the np.vectorize broadcast")


# === 4. the np.vectorize broadcast ===========================================
shapes_ok = (Kxy.shape == (rep, n, n)) and (np.asarray(kernel.gram_matrix(xs, xs)).shape ==
                                            (len(support), len(support)))
report("4. Gram matrix shapes are (rep,n,n) and (m,m)", shapes_ok,
       "" if shapes_ok else f"got {Kxy.shape} and {np.asarray(kernel.gram_matrix(xs, xs)).shape}")


# === 5. alpha assembly and its row alignment against K =======================
p1 = ms1.get_pmfs_df(kernel.support)
p2 = ms2.get_pmfs_df(kernel.support)
alpha_df = (p1 - p2).fillna(p1).fillna(-p2)

index_ok = list(alpha_df.index) == list(kernel.support)
report("5. alpha rows are in the same order as kernel.support", index_ok,
       "" if index_ok else "alpha.T @ K @ alpha is contracting mismatched rankings")

cols_ok = list(p1.columns) == list(p2.columns)
report("5b. the two pmf frames share column labels", cols_ok,
       "" if cols_ok else f"{list(p1.columns)[:5]} vs {list(p2.columns)[:5]} -- the subtraction "
                          f"is pairing the wrong repetitions")

alpha = alpha_df.values
mass_ok = np.allclose(alpha.sum(axis=0), 0) and np.allclose(p1.values.sum(axis=0), 1)
report("5c. each pmf sums to 1 and each alpha column sums to 0", mass_ok,
       "" if mass_ok else f"pmf column sums in [{p1.values.sum(axis=0).min():.4f}, "
                          f"{p1.values.sum(axis=0).max():.4f}]")


# === 6. the cached support Gram matrix =======================================
kernel.K = None
K_fresh = np.asarray(kernel.gram_matrix(xs, xs))
_ = kernel.mmd_distribution(sample, n=n, rep=10, seed=seed, disjoint=disjoint,
                            replace=replace, method="embedding",
                            use_cached_support_matrix=True)
cache_ok = np.allclose(np.asarray(kernel.K), K_fresh)
report("6. cached kernel.K matches a fresh one at the current nu", cache_ok,
       "" if cache_ok else "the cache predates the current nu or support; only set_support clears it")


# === reference: MMD computed from the definition, independently of both ======
K_support = K_fresh
ref = np.sqrt(np.abs(np.einsum("ir,ij,jr->r", alpha, K_support, alpha)))

kernel.K = None
vec = kernel.mmd_distribution(sample, n=n, rep=rep, seed=seed, disjoint=disjoint,
                              replace=replace, method="vectorized")
kernel.K = None
emb = kernel.mmd_distribution(sample, n=n, rep=rep, seed=seed, disjoint=disjoint,
                              replace=replace, method="embedding")

print("\nper-repetition max deviation from the reference")
print(f"  vectorized  {np.abs(vec - ref).max():.3e}   mean {vec.mean():.6f}")
print(f"  embedding   {np.abs(emb - ref).max():.3e}   mean {emb.mean():.6f}")
print(f"  reference                    mean {ref.mean():.6f}")

failed = [s for s, ok in results if not ok]
print("\n" + ("all stages passed -- the discrepancy is upstream of this script "
              "(check nu, and whether kernel.support matches the sample)"
              if not failed else f"first failing stage: {failed[0]}"))