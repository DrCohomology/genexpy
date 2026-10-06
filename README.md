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
  configuration of a study, the number of experiments $n^*$ needed for external validity, and the plots to inspect the
  results (`PlotManager`).
- **Demos and experiments:** a template, two case studies (categorical encoders and BIG-bench), and the experiments
  on synthetic data of the paper.

## External validity in a nutshell

An experimental study evaluates $n_a$ alternatives $a \in [n_a]$ under different experimental conditions. Their
factors are either fixed, and define a configuration, or random: the findings should generalize over the levels of
the random factors. For a fixed configuration, the result of an experiment is a random variable $X$ with values in a
space of results $\mathcal X$ (e.g., the rankings with ties $\mathcal R_{n_a}$ of the alternatives) and distribution
$P$. Given two independent samples $\mathbf X, \mathbf X' \sim P^n$, the kernel external validity (KEV) of a study of
size $n$ is

$$
\mathrm{E}_n(P, \varepsilon) := P^n \otimes P^n\left(\mathrm{MMD}(\hat P_{\mathbf X}, \hat P_{\mathbf X'}) \leq \varepsilon\right),
$$

where $\hat P_{\mathbf X}$ is the empirical distribution of $\mathbf X$ and
$\mathrm{MMD}(P, Q) = \lVert \mu_P - \mu_Q \rVert_{\mathcal H_k}$ is the maximum mean discrepancy with kernel $k$.
The kernel formalizes the research question, i.e., which features of the results are relevant, and
$\varepsilon \in \mathbb R^+$ how similar two results must be. A study is $(\alpha, \varepsilon, n)$-valid if
$\mathrm{E}_n(P, \varepsilon) \geq \alpha$, and the number of experiments needed is

$$
n^* := \min\{n : \mathrm{E}_n(P, \varepsilon) \geq \alpha\}.
$$

The $\alpha$-quantile $q_\alpha(n)$ of $\mathrm{MMD}_n := \mathrm{MMD}(\hat P_{\mathbf X}, \hat P_{\mathbf X'})$
follows the power law $\log q_\alpha(n) + \frac{1}{2} \log n = \beta_\alpha + o(1)$. From a preliminary study of $N$
results $\mathbf y$, with empirical distribution $P_N$, genexpy estimates $q_\alpha(n)$ by resampling pairs of
samples $\mathbf x, \mathbf x'$ of size $n < N/2$ from $\mathbf y$, fits $\beta_\alpha$, and predicts

$$
\log \hat n^*_N = 2 \beta_\alpha - 2 \log \varepsilon.
$$

To make $\varepsilon$ interpretable, every kernel factors as $k = g \circ d$, where $d$ is a normalized dissimilarity
and $g$ is decreasing and convex with $g(0) = 1$. genexpy derives $\varepsilon$ from a maximum expected
dissimilarity $\delta \in [0, 1]$ (`Kernel.get_eps`): with $\varepsilon(\delta) := \sqrt{2(1 - g(\delta))}$,
$\mathbb E_{P \otimes P}\, d(X, X') \leq \delta$ implies $\mathbb E_{P^{2n}} \mathrm{MMD}_n \leq \varepsilon$.

| Kernel | Research question | $k(r_1, r_2)$ | $d(r_1, r_2)$ | $g(u)$ |
|---|---|---|---|---|
| Borda $k_\mathrm{b}^{a^*, \nu}$: `BordaKernel(idx, nu)` | Is alternative $a^*$ consistently ranked the same? | $e^{-\nu \lvert b_1 - b_2 \rvert}$ | $\lvert b_1 - b_2 \rvert / (n_a - 1)$ | $e^{-\nu (n_a - 1) u}$ |
| Jaccard $k_\mathrm{j}^t$: `JaccardKernel(t)` | Are the top-$t$ alternatives consistently the same ones? | $J_t(r_1, r_2)$ | $1 - J_t(r_1, r_2)$ | $1 - u$ |
| Mallows $k_\mathrm{m}^\nu$: `MallowsKernel(nu)` | Are the alternatives ranked consistently? | $e^{-\nu n_d}$ | $n_d / \binom{n_a}{2}$ | $e^{-\nu \binom{n_a}{2} u}$ |
| RBF $k_\mathrm{RBF}^\gamma$: `RBFKernel(gamma)` | Are the performances consistent? | $e^{-\gamma \lVert \mathbf x_1 - \mathbf x_2 \rVert^2}$ | $\lVert \mathbf x_1 - \mathbf x_2 \rVert^2 / n_a$ | $e^{-\gamma n_a u}$ |

Here, $b_l = \lvert\{a \in [n_a] : r_l(a) \geq r_l(a^*)\}\rvert$ is the Borda count of $a^*$; $J_t$ is the Jaccard
similarity of the top-$t$ tiers $r_{l,[t]} = \{a \in [n_a] : r_l(a) \leq t\}$; $n_d$ is the number of discordant
pairs; and $k_\mathrm{RBF}^\gamma$ acts on vectors of scores in $[0, 1]^{n_a}$. With the default bandwidths
(`"auto"`: $\nu = 1/(n_a - 1)$ for $k_\mathrm{b}$, $\nu = 1/\binom{n_a}{2}$ for $k_\mathrm{m}$, and $\gamma = 1/n_a$
for $k_\mathrm{RBF}$), $\varepsilon(\delta) = \sqrt{2(1 - e^{-\delta})}$; for $k_\mathrm{j}$,
$\varepsilon(\delta) = \sqrt{2\delta}$. In `config.yaml`, $a^*$ is given by name (`alternative: <name>`). In genexpy,
the best rank is 0 instead of 1. The paper describes the theory in detail.

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
externally valid with respect to (random factor), together with the kernels, $\alpha$, $\delta$, and the increment
$N_0$ of the size of the preliminary studies (`sampling.sample_size`). `demos/case_studies/Template/config.yaml`
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

`validity_analysis` returns $\hat n^*_N$ for every configuration, kernel, estimation method, $\alpha$, $\delta$, and
$N = N_0, 2N_0, \dots$ With `resample=True`, every preliminary study is drawn afresh from the results of the
configuration; with `resample=False`, the study of size $N + N_0$ extends the one of size $N$, as in an actual
incremental execution of the study.

The analysis writes to the `outputs` directory (configurable in `config.yaml`):
- `nstar_resample={True,False}.parquet`: the predicted $\hat n^*_N$ (columns: the fixed factors, `kernel`, `alpha`,
  `delta`, `eps`, `method`, `N`, `nstar`, ...);
- `MMD_precomputed/` and its snapshot `preloaded_mmd__resample={True,False}.parquet`: the estimated distributions of
  $\mathrm{MMD}_n$, reused by later runs (set `load_precomputed_mmd: True`) and by the plots;
- `MMD_approximated_icdf_coefficients/` and its snapshot `preloaded_mmd_icdf_coeff.parquet`: the coefficients of the
  approximated quantile function of $\mathrm{MMD}_n$.

### Plots

`PlotManager` saves every figure in the `figures` directory, as `<name>_*.pdf`, where `<name>` is
`project_parameters.name` in `config.yaml`.

| Method | What it shows | File |
|---|---|---|
| `plot_nstar_on_alpha_delta` | $\hat n^*_N$ as a function of $\alpha$ (left) and $\delta$ (right), for every kernel: boxplots over the configurations, at the largest $N$ of each | `<name>_nstar_alpha_delta.pdf` |
| `plot_validity_on_n` | $\mathrm{E}_n(P_N, \varepsilon(\delta))$ as a function of $n$, one panel per kernel and one line per $\delta$ (median and range over the configurations); the dashed line marks $\alpha$ | `<name>_validity_on_n.pdf` |
| `plot_simulated_experimental_study` | for one configuration, one figure per kernel and one column per $N$: $\mathrm{E}_n(P_N, \varepsilon)$ as a function of $\varepsilon$ for every $n$ (top), and the $\alpha$-quantiles of $\mathrm{MMD}_n$ with the power-law fit and $\hat n^*_N$ at $\varepsilon(\delta)$ (bottom) | `<name>_simulated_study__kernel=<kernel>.pdf` |
| `plot_validity_resampling_comparison_on_n` | $\mathrm{E}_n(P_N, \varepsilon(\delta))$ as a function of $n$, from fresh vs incremental preliminary studies; needs both `validity_analysis()` and `validity_analysis(resample=False)` | `<name>_validity_resampling_on_n__delta=<delta>.pdf` |

With `resample=False`, the first three plots use the incremental preliminary studies; the first two then add
`_nested` to the file name.

### Estimation of the distribution of the MMD

`RankingKernel.mmd_distribution(sample, n, rep, method=...)` draws $n_\text{rep}$ (`rep`) pairs of samples
$\mathbf x, \mathbf x'$ of size $n$ from `sample` ($\mathbf y$) and supports:
- `"embedding"` (default, fast):
  $\mathrm{MMD}(\hat P_{\mathbf x}, \hat P_{\mathbf x'})^2 = (\mathbf p - \mathbf p')^\top K (\mathbf p - \mathbf p')$,
  where $\mathbf p, \mathbf p'$ are the empirical pmfs of $\mathbf x, \mathbf x'$ over the $n_u$ distinct results and
  $K$ is their Gram matrix;
- `"vectorized"`: the Gram matrices of every pair $\mathbf x, \mathbf x'$, in batches;
- `"naive"` (slow, reference): the kernel is evaluated on every pair of results;
- `"approximation"` (instant): closed-form approximation of the quantile function of $\mathrm{MMD}_n$ (limiting
  distribution $n \, \mathrm{MMD}_n^2 \to 2 \sum_i \lambda_i Z_i^2$, moment matching with a scaled $\chi^2$,
  Wilson–Hilferty, and Lin's approximation of the normal quantile function). The output is the quantile function at
  `rep` levels, not a sample.

The flags `disjoint` and `replace` select the resampling scheme: the default, `disjoint=True` and `replace=False`, is
the permutation scheme, in which $\mathbf x$ and $\mathbf x'$ come from disjoint halves of $\mathbf y$ and no
experiment is repeated; `disjoint=False` and `replace=True` is the bootstrap; the other two combinations are
pessimistic (`True`, `True`) and optimistic (`False`, `False`). `RBFKernel` supports `"naive"` and
`"approximation"` only.

**Runtime.** `validity_analysis` spends most of its time in `ProjectManager.estimate_mmd`, which estimates the
distribution of $\mathrm{MMD}_n$ for every even $n < N/2$ in every preliminary study of size $N$. For each $n$, both
`"embedding"` and `"vectorized"` draw $n_\text{rep}$ pairs $\mathbf x, \mathbf x'$ from $\mathbf y$, which costs
$O(n_\text{rep} N)$. `"embedding"` then costs $O(n_\text{rep}\, n_u^2)$, independently of $n$, where $n_u$ is the
number of distinct results in the configuration: their Gram matrix is computed only once per configuration and
kernel. `"vectorized"` instead computes the Gram matrices of every pair, at $O(n_\text{rep}\, n^2 n_a^2)$ for
$k_\mathrm{m}$ (less for $k_\mathrm{j}$ and $k_\mathrm{b}$). Over a preliminary study, this adds up to
$O(n_\text{rep} N (N + n_u^2))$ for `"embedding"` and $O(n_\text{rep} N^3 n_a^2)$ for `"vectorized"`, so
`"embedding"` is faster unless the configuration has far more distinct results than the preliminary study has
experiments. On a consumer laptop, the encoders case study runs in about 7 minutes from scratch.

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
  `outputs/` directory caches the distributions of $\mathrm{MMD}_n$, so that the notebook runs in about a minute.
- `demos/case_studies/BIG-bench`: 55 LLMs evaluated on the BIG-bench tasks; every combination of task and number of
  shots is a configuration, and the subtask is the random factor.

The notebooks pass the current working directory as `demo_dir`, so they must run from their own directory (the
Jupyter default).

## Experiments

| Directory | Content | How to run |
|---|---|---|
| `nstar_estimation` | Predicts $\hat n^*_N$ from samples of a uniform distribution of rankings and checks the KEV of studies of size $\hat n^*_N$ | notebook |
| `significance` | Significance tests (Friedman + post-hoc) vs external validity: `significance_vs_validity.ipynb` (toy distribution) and `significance_vs_validity_v2.py` | notebook; `python significance_vs_validity_v2.py --out figures [--quick]` |
| `random vs iterated sampling` | $\hat n^*_N$ from fresh vs incremental preliminary studies | `python random_vs_iterated_sampling_N.py`, from the Categorical Encoders demo directory |
| `mmd_methods_comparison` | Agreement (`mmd_methods_comparison.py`, `mmd_bug.py`) and runtime (`speed comparison.py`) of the MMD estimation methods | `python <script>.py` |
| `kernel tests` | Extreme values of the kernels | `python kernel_ranges_parameters.py` |

## Citing

...
