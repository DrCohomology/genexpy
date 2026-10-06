This directory contains the external validity analysis of the BIG-bench results: language models (the alternatives) evaluated on the subtasks of BIG-bench tasks, with different numbers of shots.
Every combination of task and number of shots is a configuration; the results should be externally valid with respect to the subtasks.

- `config.yaml`: configuration of the analysis.
- `validity_analysis.ipynb`: runs the analysis (`ProjectManager.validity_analysis`) and draws the plots (`PlotManager`).
- `data/bigbench_results.parquet`: the raw results, with every score of every subtask.
- `data/bigbench_results_prefiltered.parquet`: the results analyzed, with only the preferred score of every task (one row per subtask, task, number of shots, and model).
- `outputs/`: predicted n* (`nstar_resample=*.parquet`), cached distributions of the MMD (`preloaded_mmd__resample=*.parquet`, `MMD_precomputed/`), and coefficients of the approximated quantile function of the MMD (`preloaded_mmd_icdf_coeff.parquet`, `MMD_approximated_icdf_coefficients/`).
- `figures/`: the plots.
