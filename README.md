# genexpy: Generalizability of experimental studies


genexpy is a Python module for the analysis of experimental results. 
Among the rest, it implements:
- **Ranking:** definition, samples, multisamples (samples of samples).
- **Probability distributions (PDs) over rankings:** sampling, multisampling.
- **Kernels for rankings:** Borda, Jaccard, and Mallows kernels.
- **Maximum Mean Discrepancy (MMD):** for PDs over rankings and numerical data.
- **Generalizability analysis of experimental results:** based on the MMD.

Demos for the usage of genexpy are in ``genexpy/demos``.

# Generalizability

genexpy is based on the theoretical machinery described in the paper [Generalizability of Experimental Studies]().
In short, if an experimental study $S$ yields results with a distribution $\mathbb P$, its $n$-generalizability is the probability
that two independent realizations yield similar results:
$$
n\text{-Gen}(S, \varepsilon) = \text{Pr}_{X, Y \sim \mathbb P^n} (d(X, Y) \leq \varepsilon), 
$$
where $\varepsilon$ is a user-defined dissimilarity threshold and $d$ is a distance between probability distributions.

# Installation

genexpy requires Python >= 3.13, although it is likely to work with Python >= 3.10.

The latest release (and relevant dependencies) can be installed from PyPI (**not anonymous**):
```bash
pip install genexpy
```

## Anonymized repository

To install genexpy in a virtual environment:
1. Download the repo and extract the files to a directory named `genexpy`; 
2. Navigate to `genexpy`; 
3. Create a virtual environment `venv`; 
4. Activate `venv` (the command shown is for Windows only); 
5. Install `genexpy` in `venv`.

```bash 
cd genexpy                                                           
python -m venv venv                                       
venv\Scripts\activate.bat                                 
pip install .                                                
```

# Citing 
... 


