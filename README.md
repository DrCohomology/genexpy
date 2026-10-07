# genexpy: External Validity of Experimental Studies

genexpy is a Python package to assess the **external validity** of experimental studies, i.e., the extent to which
the findings of a study generalize to unseen experimental conditions (other datasets, seeds, hardware, ...).
It also answers the practical follow-up question: *how many experiments are needed for the results to be
externally valid?*

genexpy implements:
- **Rankings:** rankings with ties of the compared alternatives, stored as adjacency matrices; samples and
  multi-samples (samples of samples) of rankings; conversion of a table of experimental results into rankings.
- **Kernels:** Borda, Jaccard, and Mallows kernels for rankings, each formalizing a different research question, and
  a Gaussian (RBF) kernel for raw scores.
- **Maximum Mean Discrepancy (MMD):** fast estimation of the distribution of the MMD between two samples of results,
  and a closed-form approximation of its quantile function.
- **Probability distributions over rankings:** uniform, degenerate, spike, and empirical distributions, to simulate
  experimental results.
- **External validity analysis:** a configuration-driven pipeline (`ProjectManager`) that estimates, for every
  configuration of a study, the number of experiments `n*` needed for external validity, and the plots to inspect the
  results (`PlotManager`).
- **Demos and experiments:** a template, two case studies (categorical encoders and BIG-bench), and the experiments
  on synthetic data of the paper.

## External validity in a nutshell

An experimental study evaluates a set of alternatives under different experimental conditions. Their factors are
either fixed, and define a configuration, or random: the findings should generalize over the levels of the random
factors. For a fixed configuration, the result of an experiment is a random variable with values in a space of
results (e.g., the rankings with ties of the alternatives) and some unknown distribution.

The kernel external validity (KEV) of a study of size `n` is the probability that two independent studies of size
`n` give similar results, i.e., that the maximum mean discrepancy (MMD) between the empirical distributions of their
results is at most a tolerance `eps`. The kernel of the MMD formalizes the research question, i.e., which features of
the results are relevant, and `eps` how similar two results must be. A study is valid if its KEV is at least `alpha`,
and `n*` is the smallest study size for which this holds.

The `alpha`-quantile of the MMD between two samples of size `n` decreases approximately as `1/sqrt(n)`. From a
preliminary study of `N` results, genexpy estimates this quantile by resampling pairs of samples of size `n < N/2`,
fits the power law, and extrapolates the study size `n*` at which the quantile drops to `eps`.

To make `eps` interpretable, every kernel is a decreasing, convex function of a normalized dissimilarity between
results, with values in [0, 1]. genexpy derives `eps` from `delta`, the maximum expected dissimilarity between two
results (`Kernel.get_eps`): if the expected dissimilarity is at most `delta`, the expected MMD is at most `eps`.

| Kernel | Research question | Definition |
|---|---|---|
| Borda: `BordaKernel(idx, nu)` | Is the alternative `idx` consistently ranked the same? | `exp(-nu * abs(b1 - b2))`, where `b1`, `b2` are the Borda counts of the alternative (number of alternatives ranked no better than it) in the two rankings |
| Jaccard: `JaccardKernel(t)` | Are the top-`t` alternatives consistently the same ones? | Jaccard similarity of the sets of alternatives ranked in the top `t` tiers |
| Mallows: `MallowsKernel(nu)` | Are the alternatives ranked consistently? | `exp(-nu * nd)`, where `nd` is the number of discordant pairs |
| RBF: `RBFKernel(gamma)` | Are the performances consistent? | `exp(-gamma * squared_distance(x1, x2))`, on vectors of scores in [0, 1] |

The default bandwidths (`"auto"`) are `nu = 1/(na - 1)` for the Borda kernel, `nu = 1/(na * (na - 1) / 2)` for the
Mallows kernel, and `gamma = 1/na` for the RBF kernel, where `na` is the number of alternatives; with them,
`eps = sqrt(2 * (1 - exp(-delta)))`. For the Jaccard kernel, `eps = sqrt(2 * delta)`. In `config.yaml`, the
alternative of the Borda kernel is given by name (`alternative: <name>`). In genexpy, the best rank is 0 instead
of 1. The paper describes the theory in detail.

## Installation

genexpy requires Python >= 3.13.

### Anonymized repository

To install genexpy in a virtual environment:
1. Download the repo and extract the files to a directory named `genexpy`;
2. Navigate to `genexpy`;
3. Create a virtual environment `venv`;
4. Activate `venv`;
5. Install the dependencies and `genexpy` in `venv`.

```bash
cd genexpy
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate.bat
pip install -r requirements.txt
pip install .
```

`requirements.txt` also installs Jupyter and `scikit-posthocs`, used by the notebooks and the significance
experiments. `PlotManager` renders text with LaTeX (`text.usetex`): it needs a LaTeX installation with the
`cm-super` fonts.

## Quickstart

### Rankings, kernels, and the MMD

```python
import numpy as np
from genexpy import SampleAM, MallowsKernel, JaccardKernel, UniformDistribution

# n_a = 3 alternatives (rows) ranked under 4 experimental conditions (columns); 0 is the best rank, ties are allowed
rv = np.array([[0, 0, 1, 0],
               [1, 0, 0, 1],
               [2, 1, 2, 2]])
sample = SampleAM.from_rank_vector_matrix(rv)   # rankings stored as (bytes of) adjacency matrices
sample.to_rank_vector_matrix()                  # back to rank vectors

kernel = MallowsKernel(nu="auto", na=3)
kernel(rv[:, 0], rv[:, 1])                      # k_m(r_1, r_2)
kernel.gram_matrix(sample)                      # Gram matrix of the sample

# distribution of MMD_10 from a preliminary study of N = 100 results
y = UniformDistribution(na=5, seed=0).sample(100)
jaccard = JaccardKernel(t=1)
mmd = jaccard.mmd_distribution(y, n=10, rep=500, seed=1, method="embedding")
kev = np.mean(mmd <= jaccard.get_eps(delta=0.05))   # estimate of E_10(P_N, eps(0.05))
```

### External validity analysis of a results file

The results file is a table (parquet) with one row per alternative and experimental condition: a column for the
alternatives, a column for the result (target), and one column per experimental factor. The configuration file
`config.yaml` declares which factors are held constant, which define the configurations to analyze separately
(design factors; together with the held-constant ones, the fixed factors), and which one the results should be
externally valid with respect to (random factor), together with the kernels, `alpha`, `delta`, and the increment
`N0` of the size of the preliminary studies (`sampling.sample_size`). `demos/case_studies/Template/config.yaml`
documents every field.

```python
import os
from genexpy.managers import ProjectManager, PlotManager

pm = ProjectManager("config.yaml", demo_dir=os.getcwd())
df_nstar = pm.validity_analysis()                        # fresh preliminary studies
df_nstar_incr = pm.validity_analysis(resample=False)     # incremental preliminary studies

plotter = PlotManager("config.yaml", demo_dir=os.getcwd())
plotter.plot_nstar_on_alpha_delta(alpha_fixed=0.95, delta_fixed=0.05)
plotter.plot_validity_on_n(alpha=0.95, deltas=[0.01, 0.05, 0.1])
plotter.plot_simulated_experimental_study(configuration={"model": "DTC", "tuning": "no", "scoring": "F1"},
                                          alpha=0.95, delta=0.05)
plotter.plot_validity_resampling_comparison_on_n(alpha=0.95, delta=0.05)
```

`validity_analysis` returns the estimated `n*` for every configuration, kernel, estimation method, `alpha`, `delta`,
and preliminary-study size `N = N0, 2*N0, ...`. With `resample=True`, every preliminary study is drawn afresh from the
results of the configuration; with `resample=False`, the study of size `N + N0` extends the one of size `N`, as in an
actual incremental execution of the study.

The analysis writes to the `outputs` directory (configurable in `config.yaml`):
- `nstar_resample={True,False}.parquet`: the estimated `n*` (columns: the fixed factors, `kernel`, `alpha`,
  `delta`, `eps`, `method`, `N`, `nstar`, ...);
- `MMD_precomputed/` and its snapshot `preloaded_mmd__resample={True,False}.parquet`: the estimated distributions of
  the MMD, reused by later runs (set `load_precomputed_mmd: True`) and by the plots;
- `MMD_approximated_icdf_coefficients/` and its snapshot `preloaded_mmd_icdf_coeff.parquet`: the coefficients of the
  approximated quantile function of the MMD.

### Plots

`PlotManager` saves every figure in the `figures` directory, as `<name>_*.pdf`, where `<name>` is
`project_parameters.name` in `config.yaml`.

| Method | What it shows | File |
|---|---|---|
| `plot_nstar_on_alpha_delta` | estimated `n*` as a function of `alpha` (left) and `delta` (right), for every kernel: boxplots over the configurations, at the largest `N` of each | `<name>_nstar_alpha_delta.pdf` |
| `plot_validity_on_n` | estimated KEV as a function of `n`, one panel per kernel and one line per `delta` (median and range over the configurations); the dashed line marks `alpha` | `<name>_validity_on_n.pdf` |
| `plot_simulated_experimental_study` | for one configuration, one figure per kernel and one column per `N`: estimated KEV as a function of `eps` for every `n` (top), and the `alpha`-quantiles of the MMD with the power-law fit and the estimated `n*` (bottom) | `<name>_simulated_study__kernel=<kernel>.pdf` |
| `plot_validity_resampling_comparison_on_n` | estimated KEV as a function of `n`, from fresh vs incremental preliminary studies; needs both `validity_analysis()` and `validity_analysis(resample=False)` | `<name>_validity_resampling_on_n__delta=<delta>.pdf` |

With `resample=False`, the first three plots use the incremental preliminary studies; the first two then add
`_nested` to the file name.

### Estimation of the distribution of the MMD

`RankingKernel.mmd_distribution(sample, n, rep, method=...)` draws `rep` pairs of samples of size `n` from `sample`
and supports:
- `"embedding"` (default, fast): the squared MMD is computed from the difference of the empirical pmfs of the two
  samples over the distinct results in `sample`, and the Gram matrix of those distinct results;
- `"vectorized"`: the Gram matrices of every pair of samples, in batches;
- `"naive"` (slow, reference): the kernel is evaluated on every pair of results;
- `"approximation"` (instant): closed-form approximation of the quantile function of the MMD (limiting distribution
  as a weighted sum of chi-squared variables, moment matching with a scaled chi-squared, Wilson–Hilferty, and Lin's
  approximation of the normal quantile function). The output is the quantile function at `rep` levels, not a sample.

The flags `disjoint` and `replace` select the resampling scheme: the default, `disjoint=True` and `replace=False`, is
the permutation scheme, in which the two samples come from disjoint halves of `sample` and no experiment is repeated;
`disjoint=False` and `replace=True` is the bootstrap; the other two combinations are pessimistic (`True`, `True`) and
optimistic (`False`, `False`). `RBFKernel` supports `"naive"` and `"approximation"` only.

**Runtime.** `validity_analysis` spends most of its time in `ProjectManager.estimate_mmd`, which estimates the
distribution of the MMD for every even `n < N/2` in every preliminary study of size `N`. For each `n`, both
`"embedding"` and `"vectorized"` draw `rep` pairs of samples from the preliminary study, which costs `O(rep * N)`.
`"embedding"` then costs `O(rep * nu^2)`, independently of `n`, where `nu` is the number of distinct results in the
configuration: their Gram matrix is computed only once per configuration and kernel. `"vectorized"` instead computes
the Gram matrices of every pair, at `O(rep * n^2 * na^2)` for the Mallows kernel (less for the Jaccard and Borda
kernels), where `na` is the number of alternatives. Over a preliminary study, this adds up to
`O(rep * N * (N + nu^2))` for `"embedding"` and `O(rep * N^3 * na^2)` for `"vectorized"`, so `"embedding"` is faster
unless the configuration has far more distinct results than the preliminary study has experiments. On a consumer
laptop, the encoders case study runs in about 7 minutes from scratch.

## Package layout

```
genexpy/
├── utils/
│   ├── rankings.py    AdjacencyMatrix, UniverseAM, SampleAM, MultiSampleAM, get_matrix_from_df
│   └── relations.py   score2rv, vec2rv: scores to (dense) ranks
├── kernels/
│   ├── base.py        Kernel base class, approximation of the quantile function of the MMD
│   ├── rankings.py    RankingKernel, BordaKernel, JaccardKernel, MallowsKernel
│   └── vectors.py     VectorKernel, RBFKernel
├── random.py          UniformDistribution, DegenerateDistribution, MDegenerateDistribution,
│                      SpikeDistribution, PMFDistribution
└── managers.py        ProjectManager (external validity analysis), PlotManager (plots)
demos/case_studies/
├── Template/                  config.yaml and notebook to start a new analysis
├── Categorical Encoders/      the encoders case study of the paper
└── BIG-bench/                 the LLM case study of the paper
experiments/synthetic_data/
├── nstar_estimation/          are studies of size n* valid? (uniform distribution of rankings)
├── significance/              significance tests vs external validity
├── random vs iterated sampling/   fresh vs incremental preliminary studies
├── mmd_methods_comparison/    agreement and speed of the MMD estimation methods
└── kernel tests/              sanity checks of the kernels' ranges
```

## Demos

Every demo directory contains a `config.yaml`, a `validity_analysis.ipynb` notebook that runs the analysis and draws
the plots, and the results in `data/`; running the notebook fills `outputs/` and `figures/`.
- `demos/case_studies/Template`: copy this directory, point `config.yaml` to your results file, and run
  `validity_analysis.ipynb`. It is preconfigured for the categorical encoders data.
- `demos/case_studies/Categorical Encoders`: 32 categorical encoders evaluated on 50 datasets; every combination of
  model, tuning strategy, and scoring is a configuration (48 in total), and `dataset` is the random factor. Its
  `outputs/` directory caches the distributions of the MMD, so that the notebook runs in about a minute.
- `demos/case_studies/BIG-bench`: 55 LLMs evaluated on the BIG-bench tasks; every combination of task and number of
  shots is a configuration, and the subtask is the random factor.

The notebooks pass the current working directory as `demo_dir`, so they must run from their own directory (the
Jupyter default).

## Experiments

| Directory | Content | How to run |
|---|---|---|
| `nstar_estimation` | Estimates `n*` from samples of a uniform distribution of rankings and checks the KEV of studies of size `n*` | notebook |
| `significance` | Significance tests (Friedman + post-hoc) vs external validity: `significance_vs_validity.ipynb` (toy distribution) and `significance_vs_validity_v2.py` | notebook; `python significance_vs_validity_v2.py --out figures [--quick]` |
| `random vs iterated sampling` | Estimated `n*` from fresh vs incremental preliminary studies | `python random_vs_iterated_sampling_N.py`, from the Categorical Encoders demo directory |
| `mmd_methods_comparison` | Agreement (`mmd_methods_comparison.py`, `mmd_bug.py`) and runtime (`speed comparison.py`) of the MMD estimation methods | `python <script>.py` |
| `kernel tests` | Extreme values of the kernels | `python kernel_ranges_parameters.py` |

## Citing

...
