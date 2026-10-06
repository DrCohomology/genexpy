This directory contains the external validity analysis of the categorical encoders experimental data, which reproduces the case study of the paper.

- `config.yaml`: configuration of the analysis.
- `validity_analysis.ipynb`: runs the analysis (`ProjectManager.validity_analysis`) and draws the plots (`PlotManager`).
- `validity_analysis_old.ipynb`: an earlier, shorter version of the same notebook.
- `outputs/`: predicted n* (`nstar_resample=*.parquet`), cached distributions of the MMD (`preloaded_mmd__resample=*.parquet`, `MMD_precomputed/`), and coefficients of the approximated quantile function of the MMD (`preloaded_mmd_icdf_coeff.parquet`, `MMD_approximated_icdf_coefficients/`).
- `figures/`: the plots.
