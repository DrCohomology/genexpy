"""
Managers for the external validity analysis of experimental studies.

- ``ProjectManager`` reads a configuration file (``config.yaml``) and a file of experimental results, and estimates,
  for every configuration of the design factors, kernel, and thresholds (alpha, delta), the number of experiments n*
  needed for the results to be externally valid. The distributions of the MMD are cached on disk.
- ``PlotManager`` loads the outputs of a ``ProjectManager`` and reproduces the plots of the paper.

Example
-------
>>> pm = ProjectManager("config.yaml", demo_dir="demos/case_studies/Categorical Encoders")
>>> df_nstar = pm.validity_analysis()
>>> plotter = PlotManager("config.yaml", demo_dir="demos/case_studies/Categorical Encoders")
>>> plotter.plot_nstar_on_alpha_delta()
"""
import ast
import os
import re
import shutil
import warnings
import zlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from itertools import product
from pathlib import Path
from typing import List, Literal, Union
from tqdm.auto import tqdm

from genexpy import kernels
from genexpy.kernels.base import Kernel, mmd_spectrum_moments
from genexpy.utils import rankings as ru


def dict2str(d: dict) -> str:
    """String representation of a configuration used in file names, e.g. "dict(model='DTC', tuning='no')"."""
    s = str(d)
    s = re.sub(r"^\{(.*)}$", r"dict(\1)", s.strip())
    return re.sub(r"'(\w+)':", r"\1=", s)


def str2dict(s: str) -> dict:
    """Inverse of dict2str. Also accepts dictionary literals, e.g. "{'model': 'DTC'}"."""
    s = s.strip()
    if s.startswith("dict("):
        call = ast.parse(s, mode="eval").body
        return {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    return ast.literal_eval(s)


def configuration_mask(df: pd.DataFrame, configuration: dict) -> np.ndarray:
    """Boolean mask of the rows of df matching every factor: level pair in configuration."""
    mask = np.ones(len(df), dtype=bool)
    for factor, lvl in configuration.items():
        mask &= (df[factor] == lvl).to_numpy()
    return mask


def _to_python(x):
    """numpy scalars to python scalars, for hashing and comparisons."""
    return x.item() if isinstance(x, np.generic) else x


class ProjectManager:
    """
    External validity analysis of the experimental results described by a configuration file.

    Upon initialization, the manager reads the configuration file, loads and filters the results, converts them
    into rankings, builds the kernels, creates the output directories, and loads the cached MMD distributions.

    Parameters
    ----------
    config_yaml_path : str or Path
        Path of the configuration file, relative to demo_dir.
    demo_dir : str or Path, optional
        Directory of the project: it contains the configuration file and the outputs. Relative paths in the
        configuration file are relative to it, and it becomes the working directory. Default: the current
        working directory.
    is_project_manager : bool
        If True, create the output directories.

    Examples
    --------
    >>> pm = ProjectManager("config.yaml", demo_dir=os.getcwd())
    >>> df_nstar = pm.validity_analysis()                         # independent simulated studies
    >>> df_nstar_nested = pm.validity_analysis(resample=False)    # nested simulated studies
    """

    df_format = "parquet"

    def __init__(self, config_yaml_path: Union[str, Path], demo_dir: Union[str, Path] = None,
                 is_project_manager: bool = True):
        # --- Configuration and flags ---
        self.is_project_manager = is_project_manager
        self.demo_dir = Path(demo_dir if demo_dir is not None else os.getcwd()).resolve()
        self.config_yaml_path = self.demo_dir / config_yaml_path
        self.project_name = None
        self.load_precomputed_mmd = None
        self.dump_results = None
        self.delete_existing_results = None
        self.verbose = True
        self.seed = None

        # --- File and directory management ---
        self.figures_dir = None
        self.outputs_dir = None
        self.sample_mmd_dir = None  # contains resampled MMDs
        self.approx_mmd_dir = None  # contains CDF of the MMD

        # --- Core data structures ---
        self.estimation_methods = None
        self.dfmmd = None
        self.dfmmds = {True: None, False: None}  # precomputed MMD, keyed by the resample flag
        self._mmd_index = {}  # lookup tables of the precomputed MMD, keyed by the resample flag
        self.df_nstar = None
        self.df_nstar_nested = None
        self.icdf_coefficients = None
        self.factors_dict = None
        self.results = None
        self.results_matrix = None
        self.results_rankings = None
        self.kernels = None

        # --- Factors ---
        self.all_factors: list = []
        self.design_factors: list = []
        self.random_factors: list = []
        self.held_constant_factors: list = []
        self.fixed_factors: list = []  # design + held-constant factors

        # --- Configurations ---
        self.na = None
        self.config_kernels = None
        self.config_sampling = None
        self.config_data = None
        self.config_params = None

        # --- Precomputed data ---
        self.precomputed_configurations = set()
        self.precomputed_kernels = set()
        self.precomputed_Ns = set()
        self.precomputed_mmd_filename_pattern = (
            r"configuration=dict(\(.*?\))__kernel=([A-Za-z0-9_]+\(.*?\))__N=(\d+)(?:__resample=(True|False))?"
            r"(?:__method=(\w+))?(?:__disjoint=(True|False))?(?:__replace=(True|False))?"
        )
        self.mmd_icdf_coefficients_filename_pattern = (
            r"configuration=dict(\(.*?\))__kernel=([A-Za-z0-9_]+\(.*?\))__N=(\d+)(?:__resample=(True|False))?"
        )

        # --- Flags for experimental factors ---
        self.flag_design_factor = "_all"
        self.flag_held_constant_factor = None  # HC factors are those that are neither random nor design
        self.flag_random_factor = None

        # --- Initialization steps ---
        self._load_config_file()
        if self.verbose:
            print("[INFO] Loaded configuration file.")

        if self.is_project_manager:
            self._create_project_directories()
            if self.verbose:
                print("[INFO] Created project directories.")

        if self.load_precomputed_mmd:
            self._load_all_precomputed_mmd()

    @property
    def icdf_coefficiens(self):
        """Misspelled name of icdf_coefficients, kept for backwards compatibility."""
        return self.icdf_coefficients

    def _load_config_file(self):
        """Load experiment configuration from YAML and initialize project parameters."""

        config = self._read_yaml_config()
        self._initialize_paths(config["paths"])
        self._set_project_parameters(config["project_parameters"])
        self._validate_flags()

        self._prepare_config_data(config)
        self.config_params = self._prepare_config_params(config["parameters"])
        self.config_sampling = config["sampling"]
        self.estimation_methods = config["nstar_estimation_methods"]
        self.seed = self.config_params.get("seed")

        self._set_factor_lists()
        self.results = self._load_and_filter_dataset()
        self.config_data["factor_configurations"] = self._extract_factor_configurations()

        # rankings (rows: alternatives, sorted; columns: experimental conditions) and number of alternatives kept
        self._compute_results_matrices()
        self.na = self.results_rankings.shape[0]

        self.config_kernels = config["kernels"]
        self._load_kernels(self.config_kernels)

    # ---------------------- Subroutines ----------------------

    def _read_yaml_config(self) -> dict:
        with open(self.config_yaml_path, "r") as file:
            return yaml.safe_load(file)

    def _initialize_paths(self, paths_cfg: dict):
        try:
            if self.demo_dir != Path(os.getcwd()).resolve():
                os.chdir(self.demo_dir)
                print(f"[INFO] Moved working directory to {self.demo_dir}")
        except FileNotFoundError:
            print(f"[WARNING] Failed moving working directory to {self.demo_dir}.")

        self.outputs_dir = self.demo_dir / paths_cfg.get("outputs_dir", "outputs")
        self.figures_dir = self.demo_dir / paths_cfg.get("figures_dir", "figures")
        self.sample_mmd_dir = self.outputs_dir / "MMD_precomputed"
        self.approx_mmd_dir = self.outputs_dir / "MMD_approximated_icdf_coefficients"

    def _set_project_parameters(self, general_cfg: dict):
        self.project_name = general_cfg["name"]
        self.delete_existing_results = general_cfg["delete_existing_results"]
        self.load_precomputed_mmd = general_cfg["load_precomputed_mmd"]
        self.dump_results = general_cfg["dump_results"]
        self.verbose = general_cfg["verbose"]

    def _validate_flags(self):
        if self.delete_existing_results and self.load_precomputed_mmd:
            warnings.warn(
                "Loading precomputed MMD is not possible if results are deleted. "
                "Setting delete_existing_results to False."
            )
            self.delete_existing_results = False

    def _prepare_config_params(self, params_cfg: dict) -> dict:
        for key in ("alpha", "delta"):
            if isinstance(params_cfg[key], (int, float)):
                params_cfg[key] = [params_cfg[key]]
        params_cfg.setdefault("tol_missing_alternatives", 0.2)
        params_cfg.setdefault("tol_missing_conditions", 0.2)
        params_cfg.setdefault("Nmax", None)
        return params_cfg

    def _prepare_config_data(self, config: dict) -> None:
        if sum(value is None for value in config["data"]["experimental_factors_name_lvl"].values()) != 1:
            raise ValueError("Exactly one factor must be set to null (random factor) in config.yaml.")
        self.config_data = config["data"]

    def _load_and_filter_dataset(self) -> pd.DataFrame:
        dataset_path = Path(self.config_data["dataset_path"])
        df = pd.read_parquet(dataset_path if dataset_path.is_absolute() else self.demo_dir / dataset_path)

        # Remove the columns not indicated as either factors, col of alternatives, or col of target (in config.yaml)
        tokeep = (self.all_factors + [self.config_data["alternatives_col_name"], self.config_data["target_col_name"]])
        df = df[tokeep]

        # Filter for the held-constant factors
        held_constant = {factor: lvl for factor, lvl in self.config_data["experimental_factors_name_lvl"].items()
                         if lvl not in [self.flag_random_factor, self.flag_design_factor]}
        if len(held_constant) == 0:
            return df
        return df.loc[configuration_mask(df, held_constant)].reset_index(drop=True)

    def _extract_factor_configurations(self) -> dict:
        factors = [factor for factor, lvl in self.config_data["experimental_factors_name_lvl"].items()
                   if lvl != self.flag_random_factor]
        if len(factors) == 0:
            return {"None": self.results.index}
        return self.results.groupby(factors).groups

    def _set_factor_lists(self):
        self.all_factors = list(self.config_data["experimental_factors_name_lvl"].keys())
        self.design_factors = [
            f for f, lvl in self.config_data["experimental_factors_name_lvl"].items()
            if lvl == self.flag_design_factor
        ]
        self.random_factors = [
            f for f, lvl in self.config_data["experimental_factors_name_lvl"].items()
            if lvl == self.flag_random_factor
        ]
        self.held_constant_factors = [
            f for f, lvl in self.config_data["experimental_factors_name_lvl"].items()
            if lvl not in [self.flag_design_factor, self.flag_random_factor]
        ]
        self.fixed_factors = self.design_factors + self.held_constant_factors

    def _compute_results_matrices(self):
        """Pivot the results into matrices alternatives x conditions, of target values and of rankings."""
        kwargs = dict(factors=list(self.all_factors),
                      alternatives=self.config_data["alternatives_col_name"],
                      target=self.config_data["target_col_name"],
                      lower_is_better=self.config_data["target_is_error"],
                      impute_missing=True,
                      tol_missing_indices=self.config_params["tol_missing_alternatives"],
                      tol_missing_columns=self.config_params["tol_missing_conditions"],
                      as_numpy=False)
        self.results_matrix = ru.get_matrix_from_df(self.results, get_rankings=False, **kwargs)
        self.results_rankings = ru.get_matrix_from_df(self.results, get_rankings=True, **kwargs)

    def _load_kernels(self, kernels_cfg: list):
        """
        Instantiate one kernel per combination of the parameters in the configuration file. The alternatives are
        those kept in self.results_rankings, in the same order as its rows.
        """
        alternatives = self.results_rankings.index.to_numpy()
        self.kernels = []
        for kernel_dict in kernels_cfg:
            kernel_params = kernel_dict.get("params") or {}
            for values in product(*kernel_params.values()):
                pc = dict(zip(kernel_params.keys(), values), na=len(alternatives), ordered_alternatives=alternatives)
                self.kernels.append(Kernel.from_name_and_parameters(kernel_dict["name"], **pc))

    def _create_project_directories(self):
        if self.delete_existing_results:
            shutil.rmtree(self.outputs_dir, ignore_errors=True)

        for directory, text in [
            (self.sample_mmd_dir, "This directory contains the precomputed distributions of the MMD."),
            (self.approx_mmd_dir, "This directory contains the coefficients for the ICDF (quantile function) of the MMD."),
            (self.figures_dir, "This directory contains the figures and plots."),
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            readme = directory / "README.md"
            if not readme.exists():
                readme.write_text(text + "\n")

    def get_configurations(self, df_grouped: pd.DataFrame) -> dict:
        """Levels of the fixed (design and held-constant) factors of a dataframe holding a single configuration."""
        # Check that design and held-constant factors have been filtered correctly. They should have exactly one unique values
        if (df_grouped.nunique()[self.fixed_factors] > 1).any():
            raise ValueError("Factor levels not unique after query.")

        # Current levels of design and held-constant factor
        return dict(df_grouped[self.fixed_factors].iloc[0])

    def _get_configurations_and_grouped_df(self) -> list:
        if len(self.fixed_factors) == 0:
            return [({}, self.results)]

        df_grouped_list = [x[1] for x in self.results.groupby(self.fixed_factors)]
        configurations = [self.get_configurations(df_grouped) for df_grouped in df_grouped_list]

        return list(zip(configurations, df_grouped_list))

    @staticmethod
    def _get_query_string_from_configuration(configuration: dict) -> str:
        """pandas query string selecting a configuration (prefer configuration_mask, which handles any name)."""
        return " and ".join(
            f"`{factor}` == {repr(str(lvl)) if isinstance(lvl, str) else lvl}"
            for factor, lvl in configuration.items()
        )

    def _get_seed(self, configuration: dict, N: int):
        """
        Seed of the simulated study of size N of a configuration, derived from the seed in the configuration file.
        All kernels and estimation methods share the same simulated study. None if no seed is configured.
        """
        if self.seed is None:
            return None
        return np.random.SeedSequence([int(self.seed), int(N), zlib.crc32(dict2str(configuration).encode())])

    def _resample_indices(self, n_conditions: int, configuration: dict, N: int) -> np.ndarray:
        """Indices of the N experimental conditions (drawn with replacement) of a simulated study."""
        return np.random.default_rng(self._get_seed(configuration, N)).choice(n_conditions, size=N, replace=True)

    # ---- Routines to load and dump files
    def _load_all_precomputed_mmd(self):
        """Load the precomputed MMD for both sampling schemes, keeping them in separate dataframes."""
        for resample in (True, False):
            self.dfmmds[resample] = self._load_precomputed_mmd_df(resample=resample, verbose=self.verbose)
        self.dfmmd = self.dfmmds[True]

        if self.verbose:
            nrows = {k: 0 if v is None else len(v) for k, v in self.dfmmds.items()}
            print(f"[INFO] Loaded precomputed MMD: {nrows[True]} rows resampled, {nrows[False]} rows nested.")

    @staticmethod
    def _source_files(directory: Path, df_format: str) -> list:
        return sorted(directory.glob(f"*.{df_format}")) if directory.exists() else []

    @staticmethod
    def _snapshot_is_stale(snapshot_path: Path, directory: Path, files: list) -> bool:
        """
        A snapshot (concatenation of the files in directory) is stale if files were added, removed or rewritten after
        it was written. Without files, the snapshot is the only copy of the data and is never stale.
        """
        if len(files) == 0:
            return False
        return max(p.stat().st_mtime for p in [directory, *files]) > snapshot_path.stat().st_mtime

    def _load_preloaded_mmd_df(self, resample: bool = True) -> Union[pd.DataFrame, None]:
        preloaded_path = self.outputs_dir / f"preloaded_mmd__resample={resample}.parquet"
        if not preloaded_path.exists():
            return None

        if self._snapshot_is_stale(preloaded_path, self.sample_mmd_dir,
                                   self._source_files(self.sample_mmd_dir, self.df_format)):
            return None

        dfmmd = pd.read_parquet(preloaded_path)
        if "resample" not in dfmmd.columns or not (dfmmd["resample"] == resample).all():
            warnings.warn(f"Preloaded MMD for resample={resample} does not match its flag. Reloading from files.")
            return None

        return dfmmd

    def _filter_precomputed_df(self, df: pd.DataFrame, configuration_str: str = None, kernel_name: str = None,
                               N: int = None) -> pd.DataFrame:
        mask = np.ones(len(df), dtype=bool)
        if configuration_str is not None:
            mask &= configuration_mask(df, str2dict(f"dict{configuration_str}"))
        if kernel_name is not None:
            mask &= (df["kernel"] == kernel_name).to_numpy()
        if N is not None:
            mask &= (df["N"] == int(N)).to_numpy()
        return df.loc[mask]

    def _load_precomputed_mmd_df(self, configuration_str: str = None, kernel_name: str = None, N: int = None,
                                 resample: bool = True, verbose: bool = False) -> Union[pd.DataFrame, None]:
        """
        Load the precomputed MMD for the given sampling scheme, from the snapshot preloaded_mmd__resample=*.parquet if
        it is up to date, from the files in sample_mmd_dir otherwise (and then refresh the snapshot).
        configuration_str is the part in brackets of the configuration in the file names, e.g. "(model='DTC')".
        """
        preloaded_path = self.outputs_dir / f"preloaded_mmd__resample={resample}.parquet"
        unfiltered = configuration_str is None and kernel_name is None and N is None

        dfmmd = self._load_preloaded_mmd_df(resample)
        if dfmmd is not None:
            if verbose:
                print(f"[INFO] Loaded preloaded mmd dataframe for resample={resample}.")
            return dfmmd if unfiltered else self._filter_precomputed_df(dfmmd, configuration_str, kernel_name, N)

        dfs = []
        for filepath in self._source_files(self.sample_mmd_dir, self.df_format):
            match self.df_format:
                case "parquet":
                    matched = re.search(self.precomputed_mmd_filename_pattern, filepath.name)
                    if matched is None:
                        raise ValueError(
                            f"File name {str(filepath)} is not in a valid pattern for the precomputed MMD files. "
                            f"The accepted patterns are {self.precomputed_mmd_filename_pattern}.")
                    matched_configuration, matched_kernel, matched_N, matched_resample = matched.group(1, 2, 3, 4)

                    # files dumped before the resample field was introduced are resampled
                    matched_resample = matched_resample if matched_resample is not None else "True"

                    self.precomputed_configurations.add(matched_configuration)
                    self.precomputed_kernels.add(matched_kernel)
                    self.precomputed_Ns.add(matched_N)

                    if matched_resample != str(resample):
                        continue
                    if configuration_str is not None and matched_configuration != configuration_str:
                        continue
                    if kernel_name is not None and matched_kernel != kernel_name:
                        continue
                    if N is not None and int(matched_N) != int(N):
                        continue

                    df = pd.read_parquet(filepath)
                    # the file name is authoritative: legacy files have no resample column
                    df["resample"] = resample
                    dfs.append(df)

                case _:
                    raise NotImplementedError()
        if not dfs:
            if verbose:
                print(f"[INFO] No precomputed MMD to load for resample={resample}.")
            return None

        dfmmd = pd.concat(dfs, ignore_index=True)

        if verbose:
            print(f"[INFO] Loaded precomputed MMD for {len(self.precomputed_configurations)} configurations, "
                  f"{len(self.precomputed_kernels)} kernels, and {len(self.precomputed_Ns)} values of N.")

        # Filtered loads are partial: caching them would hide the other files at the next load
        if unfiltered:
            dfmmd.to_parquet(preloaded_path)
            if verbose:
                print(f"[INFO] Dumped preloaded MMD dataframe in {preloaded_path}")

        return dfmmd

    def _load_mmd_icdf_coefficients_df(self, configuration_str: str = None, kernel_name: str = None, N: int = None,
                                       verbose: bool = False) -> Union[pd.DataFrame, None]:
        """Load the coefficients (L1, L4) of the approximated quantile function of the MMD into icdf_coefficients."""
        preloaded_path = self.outputs_dir / "preloaded_mmd_icdf_coeff.parquet"
        unfiltered = configuration_str is None and kernel_name is None and N is None
        files = self._source_files(self.approx_mmd_dir, self.df_format)

        if preloaded_path.exists() and not self._snapshot_is_stale(preloaded_path, self.approx_mmd_dir, files):
            df = pd.read_parquet(preloaded_path)
            self.icdf_coefficients = df if unfiltered else self._filter_precomputed_df(df, configuration_str,
                                                                                         kernel_name, N)
            if verbose:
                print("[INFO] Loaded preloaded mmd icdf coefficients dataframe.")
            return self.icdf_coefficients

        dfs = []
        for filepath in files:
            match self.df_format:
                case "parquet":
                    matched = re.search(self.mmd_icdf_coefficients_filename_pattern, filepath.name)
                    if matched is None:
                        raise ValueError(
                            f"File name {str(filepath)} is not in a valid pattern for the MMD ICDF coefficients files. "
                            f"The accepted patterns are {self.mmd_icdf_coefficients_filename_pattern}.")
                    matched_configuration, matched_kernel, matched_N = matched.group(1, 2, 3)

                    if configuration_str is not None and matched_configuration != configuration_str:
                        continue
                    if kernel_name is not None and matched_kernel != kernel_name:
                        continue
                    if N is not None and int(matched_N) != int(N):
                        continue

                    dfs.append(pd.read_parquet(filepath))

                case _:
                    raise NotImplementedError()

        if not dfs:
            self.icdf_coefficients = None
            if verbose:
                print("[INFO] No MMD ICDF coefficients to load.")
            return None

        self.icdf_coefficients = pd.concat(dfs, ignore_index=True)
        if verbose:
            print("[INFO] Loaded MMD ICDF coefficients.")

        if unfiltered:
            self.icdf_coefficients.to_parquet(preloaded_path)
            if verbose:
                print(f"[INFO] Dumped preloaded MMD ICDF coefficients in {preloaded_path}")

        return self.icdf_coefficients

    def _load_nstar_df(self, force=True):
        if self.df_nstar is not None and not force:
            return

        match self.df_format:
            case "parquet":
                try:
                    self.df_nstar = pd.read_parquet(self.outputs_dir / f"nstar_resample=True.{self.df_format}")
                except FileNotFoundError:
                    print("[INFO] No dataframe found for resampled results.")
                try:
                    self.df_nstar_nested = pd.read_parquet(self.outputs_dir / f"nstar_resample=False.{self.df_format}")
                except FileNotFoundError:
                    print("[INFO] No dataframe found for nested (non-resampled) results.")
            case _:
                raise NotImplementedError()

    def _dump_sample_mmd_df(self, df: pd.DataFrame):
        first = df.iloc[0]
        configuration = dict2str(self.get_configurations(df))
        match self.df_format:
            case "parquet":
                df.to_parquet(
                    self.sample_mmd_dir / f"mmd__configuration={configuration}__kernel={first['kernel']}"
                                          f"__N={first['N']}__resample={first['resample']}__method={first['method']}"
                                          f"__disjoint={first['disjoint']}__replace={first['replace']}.parquet")
            case _:
                raise NotImplementedError()

    def _dump_mmd_icdf_coefficients_df(self, df: pd.DataFrame):
        first = df.iloc[0]
        configuration = dict2str(self.get_configurations(df))
        resample = f"__resample={first['resample']}" if "resample" in df.columns else ""
        match self.df_format:
            case "parquet":
                df.to_parquet(
                    self.approx_mmd_dir / f"mmd_icdf__configuration={configuration}__kernel={first['kernel']}"
                                          f"__N={first['N']}{resample}.parquet")
            case _:
                raise NotImplementedError()

    def _dump_nstar_df(self, resample: bool = True):
        if self.df_nstar is None and self.df_nstar_nested is None:
            warnings.warn("No df_nstar to dump. Run validity_analysis first to initialize it.")
            return

        match self.df_format:
            case "parquet":
                dumped = []
                for flag, df in ((True, self.df_nstar), (False, self.df_nstar_nested)):
                    if df is not None:
                        path = self.outputs_dir / f"nstar_resample={flag}.{self.df_format}"
                        df.to_parquet(path)
                        dumped.append(str(path))
            case _:
                raise NotImplementedError()

        if self.verbose:
            print(f"[INFO] Predicted nstar stored in {', '.join(dumped)}.")

    def _precomputed_mmd_lookup(self, resample: bool) -> tuple:
        """(dataframe, key columns, {key: row positions}) of the precomputed MMD, built once per dataframe."""
        dfmmd = self.dfmmds.get(resample)
        cached = self._mmd_index.get(resample)
        if cached is None or cached[0] is not dfmmd:
            key_cols = [c for c in ["kernel", "N", "method", "disjoint", "replace"] + self.fixed_factors
                        if c in dfmmd.columns]
            groups = dfmmd.groupby(key_cols, sort=False, dropna=False).indices
            index = {tuple(_to_python(v) for v in (key if isinstance(key, tuple) else (key,))): pos
                     for key, pos in groups.items()}
            cached = (dfmmd, key_cols, index)
            self._mmd_index[resample] = cached
        return cached

    def _get_existing_precomputed_mmd(self, configuration: dict, kernel_obj: kernels.base.Kernel, N: int,
                                      resample: bool = True, method: str = None) -> pd.DataFrame:
        """Precomputed MMD for a configuration, kernel, N, estimation method, and the configured sampling scheme."""
        dfmmd = self.dfmmds.get(resample)
        if dfmmd is None:
            return pd.DataFrame()

        criteria = dict(configuration, kernel=str(kernel_obj), N=N, method=method,
                        disjoint=self.config_sampling["disjoint"], replace=self.config_sampling["replace"])
        _, key_cols, index = self._precomputed_mmd_lookup(resample)
        if set(configuration) == set(self.fixed_factors) and method is not None:
            key = tuple(_to_python(criteria[c]) for c in key_cols)
            return dfmmd.iloc[index.get(key, [])]

        # general case: boolean masks on the criteria that are not None
        criteria = {k: v for k, v in criteria.items() if v is not None and k in dfmmd.columns}
        return dfmmd.loc[configuration_mask(dfmmd, criteria)]

    # ---- Routines to compute/estimate/approximate the MMD
    def _estimate_mmd_from_experiments_rankings(self, sample: ru.SampleAM, configuration: dict,
                                                kernel_obj: kernels.rankings.RankingKernel, N: int,
                                                method: Literal["naive", "vectorized", "embedding", "approximation"],
                                                resample: bool = True):

        precomputed_mmd = self._get_existing_precomputed_mmd(configuration, kernel_obj, N, resample, method)
        if not precomputed_mmd.empty:
            return precomputed_mmd

        # Get a simulated study of size N
        if resample:
            sample = ru.SampleAM(np.asarray(sample)[self._resample_indices(len(sample), configuration, N)])

        dfmmd = kernel_obj.mmd_distribution_many_n(sample=sample, nmin=2, nmax=N // 2, step=1,
                                                   rep=self.config_params["rep"],
                                                   disjoint=self.config_sampling["disjoint"],
                                                   replace=self.config_sampling["replace"],
                                                   method=method,
                                                   N=N, use_cached_support_matrix=True)

        dfmmd.loc[:, "resample"] = resample
        for factor, lvl in configuration.items():
            dfmmd.loc[:, factor] = lvl

        if self.dump_results:
            self._dump_sample_mmd_df(dfmmd)

        return dfmmd

    def _estimate_mmd_from_experiments_vectors(self, s: np.ndarray[float], configuration: dict,
                                                kernel_obj: kernels.vectors.VectorKernel, N: int,
                                                method: Literal["naive", "embedding", "approximation"],
                                                seed: int = None, resample: bool = True):

        precomputed_mmd = self._get_existing_precomputed_mmd(configuration, kernel_obj, N, resample, method)
        if not precomputed_mmd.empty:
            return precomputed_mmd

        # Get a simulated study of size N
        if resample:
            if seed is not None:
                s = np.random.default_rng(seed=seed).choice(s.T, size=N).T
            else:
                s = s[:, self._resample_indices(s.shape[1], configuration, N)]

        dfmmd = kernel_obj.mmd_distribution_many_n(s=s, nmin=2, nmax=N // 2, step=1,
                                                   rep=self.config_params["rep"],
                                                   disjoint=self.config_sampling["disjoint"],
                                                   replace=self.config_sampling["replace"],
                                                   method=method,
                                                   N=N, use_cached_support_matrix=True)

        dfmmd.loc[:, "resample"] = resample
        for factor, lvl in configuration.items():
            dfmmd.loc[:, factor] = lvl

        if self.dump_results:
            self._dump_sample_mmd_df(dfmmd)

        return dfmmd

    def estimate_mmd(self, sample, configuration: dict, kernel_obj: kernels.base.Kernel, N: int, method: str,
                     resample: bool = True) -> pd.DataFrame:
        """
        Distribution of the MMD between pairs of subsamples of size n = 2, 4, ..., N // 2 - 1 of a simulated
        study of size N (from the cache, if available).

        Parameters
        ----------
        sample : ru.SampleAM or np.ndarray
            The results of the configuration: rankings for kernels for rankings, an array (na, n_conditions) of
            target values for kernels for vectors.
        configuration : dict
            Levels of the fixed factors, stored in the output.
        kernel_obj : Kernel
            The kernel.
        N : int
            Size of the simulated study.
        method : str
            MMD estimation method, see RankingKernel.mmd_distribution.
        resample : bool
            If True, the simulated study is drawn with replacement from `sample`; if False, `sample` is the study.

        Returns
        -------
        pd.DataFrame
            Columns: n, mmd, method, N, disjoint, replace, kernel, resample, and the fixed factors.
        """
        if isinstance(sample, ru.SampleAM) and isinstance(kernel_obj, kernels.rankings.RankingKernel):
            return self._estimate_mmd_from_experiments_rankings(sample, configuration, kernel_obj, N, method, resample)
        elif isinstance(sample, np.ndarray) and isinstance(kernel_obj, kernels.vectors.VectorKernel):
            return self._estimate_mmd_from_experiments_vectors(sample, configuration, kernel_obj, N, method,
                                                               resample=resample)
        else:
            raise TypeError(f"Parameter sample with type {type(sample)} is not a valid input type.")

    def _estimate_nstar_from_experiments(self, sample: Union[ru.SampleAM, np.ndarray[float]],
                                         configuration: dict,
                                         kernel_obj: kernels.base.Kernel, N: int,
                                         method: Literal["naive", "vectorized", "embedding"] = "embedding",
                                         resample: bool = True):
        """
        n* from the alpha-quantiles q_alpha(n) of the estimated distribution of the MMD, assuming
        log(n) = -2 log(q_alpha(n)) + b0, i.e., n* = exp(b0) / eps^2.
        """

        dfmmd = self.estimate_mmd(sample, configuration, kernel_obj, N, method, resample)

        dfq = (dfmmd.groupby("n")["mmd"].quantile(self.config_params["alpha"], interpolation="higher")
               .rename("q_alpha").rename_axis(index=["n", "alpha"]).reset_index())

        if (dfq["q_alpha"] == 0.0).all():
            print(
                f"[WARNING] Degenerate quantiles for configuration: {dict2str(configuration)} and kernel: {kernel_obj}. "
                f"Skipping nstar prediction and setting nstar=1.")

        out = []
        for alpha, dftmp in dfq.groupby("alpha"):
            if (dftmp["q_alpha"] == 0.0).any():
                b0 = b1 = 0
            else:
                logq = np.log(dftmp["q_alpha"].values.reshape(-1, 1))
                logn = np.log(dftmp["n"].values.reshape(-1, 1))

                # logn = -2 * logq + b0
                b1 = -2
                b0 = np.mean(logn - b1 * logq)

            for delta in self.config_params["delta"]:
                eps = kernel_obj.get_eps(delta, na=self.na)

                nstar = np.exp(b1 * np.log(eps) + b0)

                result_dict = dict(configuration,
                                   **dict(kernel=str(kernel_obj), alpha=alpha, eps=eps, delta=delta,
                                          disjoint=self.config_sampling["disjoint"],
                                          replace=self.config_sampling["replace"],
                                          method=method, N=N, nstar=nstar))
                out.append(result_dict)
        return out

    # --- Approximation of the MMD CDF

    @staticmethod
    def wilson_hilferty(y: np.array, k: float, a: float) -> np.ndarray[float]:
        """
        Map a variable distributed as a * chi_squared(df=k) to an (approximately) standard normal one.
        """
        return ((y / (a * k)) ** (1 / 3) - (1 - 2 / (9 * k))) / np.sqrt(2 / (9 * k))

    @staticmethod
    def wilson_hilferty_inv(z: np.ndarray, k: float, a: float) -> np.ndarray[float]:
        """Inverse of wilson_hilferty."""
        return a * k * (np.sqrt(2 / (9 * k)) * z + 1 - 2 / (9 * k)) ** 3

    @staticmethod
    def normal_cdf_Lin(x: np.ndarray):
        """Lin (1989)'s approximation of the CDF of the standard normal."""
        return 1 - 0.5 * np.exp(-0.717 * x - 0.416 * x ** 2)

    @staticmethod
    def normal_icdf_Lin(alpha: np.ndarray[float]) -> np.ndarray[float]:
        """Inverse of normal_cdf_Lin."""
        c0 = -0.861779
        c1 = 0.00120192
        c2 = 514089
        c3 = 1.664 * 10 ** 6
        return c0 + c1 * np.sqrt(c2 - c3 * np.log(-2 * (alpha - 1)))

    def _mmd_cdf_approximation(self, eps: np.ndarray, L1: float, L4: float, n: int) -> float:
        """
        Evaluate the approximated CDF of the MMD.
        Close-formula approximation via the following steps:
            1. approximate the MMD_n with Q, n times a sum of chi squares (limiting distribution, see Gretton 2012)
            2. approximate Q with Y distributed as a times a chi-squared with k degrees of freedom (moment matching,
                see Solomon and Stephens 1977 )
            3. approximate Y with Z, a normal variable (Wilson Hilferty method)
            4. approximate the cdf of Z (see Lin 1989)

        """

        a = (L4 - L1 ** 2) / L1
        k = 2 * L1 ** 2 / (L4 - L1 ** 2)

        return self.normal_cdf_Lin(self.wilson_hilferty(n * eps ** 2, k, a))

    def _mmd_icdf_approximation(self, alpha: np.ndarray[float], L1, L4, n) -> np.ndarray[float]:
        """Approximated quantile function of the MMD, inverse of _mmd_cdf_approximation."""
        a = (L4 - L1 ** 2) / L1
        k = 2 * L1 ** 2 / (L4 - L1 ** 2)

        return np.sqrt(self.wilson_hilferty_inv(self.normal_icdf_Lin(alpha), k, a) / n)

    def _get_nstar_mmd_icdf_approximation(self, eps: float, alpha: float, L1: float, L4: float) -> float:
        """
        log(n*) from the approximated quantile function of the MMD: the smallest n with q_alpha(n) <= eps.

        Parameters
        ----------
        eps : float
            MMD threshold.
        alpha : float
            Desired probability.
        L1, L4 : float
            Coefficients of the approximation (see ProjectManager._estimate_nstar_from_approximation).

        Returns
        -------
        float
            log(n*).
        """
        # constant and dof of chi square, see Solomon et Stephens (1977)
        # r = 1  # here just for consistency with the source, where they do not fix it
        a = (L4 - L1 ** 2) / L1
        k = 2 * L1 ** 2 / (L4 - L1 ** 2)

        # coefficients for the ICDF of a normal approximated with Lin (1989), Choudhury (2007)
        c0 = -0.861779
        c1 = 0.00120192
        c2 = 514089
        c3 = 1.664 * 10 ** 6

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            out = -2 * np.log(eps) \
                  + 3 * np.log(np.sqrt(2 / (9 * k)) * (c0 + c1 * np.sqrt(c2 - c3 * np.log(2 * (1 - alpha)))) +
                               (1 - 2 / (9 * k))) \
                  + np.log(a * k)

        return out

    # --- Approximation of nstar

    def _estimate_nstar_from_approximation(self, sample: Union[ru.SampleAM, np.ndarray[float]], configuration: dict,
                                           kernel_obj: kernels.base.Kernel, N: int = None, resample: bool = True):
        """
        n* from the close-form approximation of the quantile function of the MMD (see kernels.base

        L1 and L4 = L1^2 + 2 L2 are computed from the sums L1 and L2 of the eigenvalues, and of their squares, of the
        kernel centered w.r.t. the empirical distribution of the (simulated) study.

        Literature
        ----------
        Lin (1989): Lin, Jinn‐Tyan. "Approximating the normal tail probability and its inverse for use on a pocket calculator." Journal of the Royal Statistical Society: Series C (Applied Statistics) 38.1 (1989): 69-70.
        Choudhury (2007) : Choudhury, Amit, Subhasis Ray, and Pradipta Sarkar. "Approximating the cumulative distribution function of the normal distribution." Journal of Statistical Research 41.1 (2007): 59-67.
        """

        alphas = np.array(self.config_params["alpha"])
        deltas = np.array(self.config_params["delta"])

        if np.min(alphas) < 0.6:
            warnings.warn("The approximation of the MMD might not be accurate for alpha < 0.6.")

        if isinstance(sample, ru.SampleAM) and isinstance(kernel_obj, kernels.rankings.RankingKernel):
            if N is not None and resample:
                sample = ru.SampleAM(np.asarray(sample)[self._resample_indices(len(sample), configuration, N)])

            support, pmf = sample.get_support_pmf()
            x = kernel_obj._convert_sample_to_input_format(support)
            K = kernel_obj.gram_matrix(x, x)

        elif isinstance(sample, np.ndarray) and isinstance(kernel_obj, kernels.vectors.VectorKernel):
            if N is not None and resample:
                sample = sample[:, self._resample_indices(sample.shape[1], configuration, N)]

            support, counts = np.unique(sample, axis=1, return_counts=True)
            pmf = counts / np.sum(counts)
            K = kernel_obj.gram_matrix(support, support)

        else:
            raise TypeError(f"Parameter sample of type {type(sample)} is not a valid input type.")

        L1, L2 = mmd_spectrum_moments(K, pmf)
        L4 = L1 ** 2 + 2 * L2

        if np.isclose(L1, 0.0):
            print(
                f"[WARNING] Degenerate operator Th for configuration: {dict2str(configuration)} and kernel: {kernel_obj}. "
                f"Skipping nstar approximation and setting nstar=1.")

        out = []
        invalid_nstar = False
        for alpha, delta in product(alphas, deltas):
            eps = kernel_obj.get_eps(delta, na=self.na)
            if not np.isclose(L1, 0.0):
                nstar_log = self._get_nstar_mmd_icdf_approximation(eps, alpha, L1, L4)

                if not np.isfinite(nstar_log):
                    nstar = np.nan
                    invalid_nstar = True
                else:
                    nstar = np.exp(nstar_log)
            else:
                nstar = 1

            result_dict = dict(configuration,
                               **dict(kernel=str(kernel_obj), alpha=alpha, eps=eps, delta=delta,
                                      disjoint=None, replace=None,
                                      method="approximation", N=N, nstar=nstar))
            out.append(result_dict)

        if self.verbose and invalid_nstar:
            print(f"[WARNING] Invalid approximation for configuration: {dict2str(configuration)} "
                  f"and kernel: {kernel_obj}. There might not be enough data.")

        if self.dump_results:
            coefficients = dict(configuration, **dict(kernel=str(kernel_obj), N=N, L1=L1, L4=L4, resample=resample))
            self._dump_mmd_icdf_coefficients_df(pd.DataFrame(coefficients, index=[0]))

        return out

    def estimate_nstar(self, sample: Union[ru.SampleAM, np.ndarray[float]], configuration: dict,
                       kernel_obj: kernels.base.Kernel, *args,
                       method: Literal["naive", "vectorized", "embedding", "approximation"] = "embedding",
                       out: List = None, **kwargs) -> List[dict]:
        """
        Estimate n* for one configuration and kernel, for every alpha and delta in the configuration file.

        Parameters
        ----------
        sample : ru.SampleAM or np.ndarray
            The results of the configuration (rankings, or an array (na, n_conditions) for kernels for vectors).
        configuration : dict
            Levels of the fixed factors, stored in the output.
        kernel_obj : Kernel
            The kernel.
        method : {"naive", "vectorized", "embedding", "approximation"}
            "approximation" uses the close-form approximation of the quantile function of the MMD; the other methods
            fit the quantiles of the estimated distribution of the MMD (see RankingKernel.mmd_distribution).
        out : list, optional
            List the results are appended to.
        *args, **kwargs :
            N (size of the simulated study) and resample (see estimate_mmd).

        Returns
        -------
        list of dict
            `out`, extended with one record (configuration, kernel, alpha, eps, delta, disjoint, replace, method,
            N, nstar) per alpha and delta.
        """

        out = out if out is not None else []
        if method == "approximation":
            tmp = self._estimate_nstar_from_approximation(sample, configuration, kernel_obj, *args, **kwargs)
        elif method in ["naive", "vectorized", "embedding"]:
            tmp = self._estimate_nstar_from_experiments(sample, configuration, kernel_obj, *args, method=method,
                                                        **kwargs)
        else:
            raise ValueError(f"Parameter method={method} is not a valid option. "
                             f"Valid options are 'naive', 'vectorized', 'embedding', 'approximation'.")

        return out.extend(tmp) or out

    def _get_Ns(self, n_conditions: int) -> range:
        """Sizes N of the simulated studies: multiples of sampling.sample_size below min(n_conditions, Nmax)."""
        Nstep = self.config_sampling["sample_size"]
        Nmax = self.config_params["Nmax"]
        Nmax = n_conditions if Nmax is None else min(n_conditions, int(Nmax))
        return range(Nstep, Nmax, Nstep)

    def _estimate_nstar_one_sample(self, kernel_obj, sample_rankings: ru.SampleAM, sample_vectors: np.ndarray,
                                   configuration: dict, N: int, out: List, resample: bool) -> List:
        if isinstance(kernel_obj, kernels.rankings.RankingKernel):
            for method in self.estimation_methods["rankings"]:
                out = self.estimate_nstar(sample=sample_rankings, configuration=configuration,
                                          kernel_obj=kernel_obj, method=method, out=out, N=N, resample=resample)
        elif isinstance(kernel_obj, kernels.vectors.VectorKernel):
            for method in self.estimation_methods["vectors"]:
                out = self.estimate_nstar(sample=sample_vectors, configuration=configuration,
                                          kernel_obj=kernel_obj, method=method, out=out, N=N, resample=resample)
        else:
            raise TypeError(f"Parameter kernel_obj with type {type(kernel_obj)} is invalid. Valid inputs are "
                            f"kernels.vector.VectorKernel or kernels.rankings.RankingKernel")
        return out

    def _validity_analysis_one_configuration(self, sample_rankings: ru.SampleAM,
                                             sample_vectors: np.ndarray[float], out: List = None,
                                             configuration: dict = None):
        """
        n* for every kernel and every size N of the simulated studies of one configuration. Every simulated study
        is drawn independently, with replacement, from the results of the configuration.
        """

        configuration = configuration if configuration is not None else dict()
        out = out if out is not None else []
        support = sample_rankings.get_support_pmf()[0]

        # Loop over the kernels
        for kernel_obj in self.kernels:
            # Set the Universe
            kernel_obj.set_support(support)

            for N in self._get_Ns(len(sample_rankings)):
                out = self._estimate_nstar_one_sample(kernel_obj, sample_rankings, sample_vectors, configuration, N,
                                                      out, resample=True)

        return out

    def _validity_analysis_one_configuration_nested(self, sample_rankings: ru.SampleAM,
                                                    sample_vectors: np.ndarray[float], out: List = None,
                                                    configuration: dict = None, seed: int = None):
        """
        As _validity_analysis_one_configuration, but with nested subsamples: the subsample of size N + Nstep
        keeps the N experiments already drawn and adds Nstep new ones, without replacement.
        """

        configuration = configuration if configuration is not None else dict()
        out = out if out is not None else []

        seed = seed if seed is not None else self._get_seed(configuration, 0)
        order = np.random.default_rng(seed=seed).permutation(len(sample_rankings))
        support = sample_rankings.get_support_pmf()[0]

        # Loop over the kernels
        for kernel_obj in self.kernels:
            # Set the Universe
            kernel_obj.set_support(support)

            for N in self._get_Ns(len(sample_rankings)):
                idx = order[:N]
                out = self._estimate_nstar_one_sample(kernel_obj, ru.SampleAM(np.asarray(sample_rankings)[idx]),
                                                      sample_vectors[:, idx], configuration, N, out, resample=False)

        return out

    def validity_analysis(self, resample: bool = True) -> pd.DataFrame:
        """
        External validity analysis: estimate n* for every configuration of the fixed factors, kernel, size N of the
        simulated study, estimation method, alpha, and delta.

        Parameters
        ----------
        resample : bool
            If True, the simulated studies of size N are drawn independently, with replacement, from the results.
            If False, they are nested: the study of size N + sample_size extends the one of size N.

        Returns
        -------
        pd.DataFrame
            One row per configuration, kernel, alpha, delta, method, and N; the column nstar holds the predicted
            number of experiments (rounded up). Also stored in self.df_nstar (resample=True) or
            self.df_nstar_nested (resample=False), and dumped to outputs_dir if dump_results.
        """

        self._compute_results_matrices()

        if self.verbose:
            na_tmp = self.results.nunique()[self.config_data['alternatives_col_name']]
            print(f"[INFO] Kept {self.results_rankings.shape[0]} / {na_tmp} indices (alternatives) and "
                  f"{self.results_rankings.shape[1]} / {len(self.results.groupby(self.all_factors))} columns (conditions).")
            print(f"[INFO] Starting the external validity analysis.")

        if self.verbose:
            iterator = tqdm(self._get_configurations_and_grouped_df(),
                            position=0, desc="Configurations", leave=True)
        else:
            iterator = self._get_configurations_and_grouped_df()

        out = []
        for configuration, _ in iterator:
            mask = np.ones(self.results_rankings.shape[1], dtype=bool)
            for col_level, value in configuration.items():
                mask &= (self.results_rankings.columns.get_level_values(col_level) == value)

            rankings = self.results_rankings.loc[:, mask]
            sample_rankings = ru.SampleAM.from_rank_vector_matrix(rankings.values)
            sample_vectors = self.results_matrix.loc[:, mask].values

            if resample:
                out = self._validity_analysis_one_configuration(sample_rankings=sample_rankings,
                                                                sample_vectors=sample_vectors, out=out,
                                                                configuration=configuration)
            else:
                out = self._validity_analysis_one_configuration_nested(sample_rankings=sample_rankings,
                                                                       sample_vectors=sample_vectors, out=out,
                                                                       configuration=configuration)

        df_out = pd.DataFrame(out)
        df_out["nstar"] = np.ceil(df_out["nstar"])

        if resample:
            self.df_nstar = df_out
        else:
            self.df_nstar_nested = df_out

        if self.dump_results:
            self._dump_nstar_df(resample)

        return df_out


class PlotManager(ProjectManager):
    """
    Plots of the outputs of a ProjectManager: predicted n*, external validity as a function of n, and simulated
    experimental studies.

    Upon initialization, the PlotManager loads the configuration file and the files created by
    ProjectManager.validity_analysis (n*, precomputed MMD, and coefficients of the approximated ICDF of the MMD).

    Parameters
    ----------
    config_yaml_path : str or Path
        Path of the configuration file, relative to demo_dir.
    demo_dir : str or Path, optional
        Directory of the project. Default: the current working directory.
    save : bool
        If True, figures are saved in figures_dir.
    show : bool
        If True, figures are shown.

    Examples
    --------
    >>> plotter = PlotManager("config.yaml", demo_dir=os.getcwd())
    >>> plotter.plot_nstar_on_alpha_delta(alpha_fixed=0.95, delta_fixed=0.05)
    >>> plotter.plot_validity_on_n(alpha=0.95, deltas=[0.01, 0.05, 0.1])
    """

    validity_symbol = r"\text{E}"  # LaTeX symbol of external validity in the plots

    def __init__(self, config_yaml_path: Union[str, Path], demo_dir: Union[str, Path] = None, save: bool = True,
                 show: bool = True):
        super().__init__(config_yaml_path, is_project_manager=False, demo_dir=demo_dir)

        self.show = show
        self.save = save
        self.boxplot_args = None
        self.pretty_kernels = None
        self.pretty_columns = None
        self.df_nstar = None
        self.df_nstar_nested = None

        if self.save:
            self.figures_dir.mkdir(parents=True, exist_ok=True)

        self._load_nstar_df()
        self._add_Nmax_column_to_dfnstar()
        self._add_latex_column_to_dfnstar()

        # precomputed MMD: already loaded by ProjectManager if load_precomputed_mmd
        for resample in (True, False):
            if self.dfmmds.get(resample) is None:
                self.dfmmds[resample] = self._load_precomputed_mmd_df(resample=resample, verbose=self.verbose)
        available = [resample for resample in (True, False) if self.dfmmds[resample] is not None]
        if available:
            self.set_resample(available[0])
        else:
            self.dfmmd = None
            warnings.warn("No precomputed MMD found: run ProjectManager.validity_analysis first. The plots that need "
                          "the distribution of the MMD are unavailable.")

        self._load_mmd_icdf_coefficients_df()
        self._load_preconfigured_plotting_parameters()

    def set_resample(self, resample: bool = True):
        """Choose which MMD dataframe (resampled or nested) the plots use."""
        if self.dfmmds.get(resample) is None:
            raise ValueError(f"No precomputed MMD with resample={resample}. "
                             f"Available: {[k for k, v in self.dfmmds.items() if v is not None]}.")
        self.dfmmd = self.dfmmds[resample]
        if self.verbose:
            print(f"[INFO] Loaded MMD for resample={resample}.")

    def _get_kernel(self, kernel_name: str) -> Kernel:
        """The kernel of the configuration file with this name (exact parameters), parsed from the name otherwise."""
        for kernel_obj in self.kernels:
            if str(kernel_obj) == kernel_name:
                return kernel_obj
        return Kernel.from_string(kernel_name)

    def _add_Nmax(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return None
        if len(self.fixed_factors) == 0:
            return df.assign(Nmax=df["N"].max())
        return df.join(df.groupby(self.fixed_factors)["N"].max(), on=self.fixed_factors, rsuffix="max")

    def _add_Nmax_column_to_dfnstar(self):
        self.df_nstar = self._add_Nmax(self.df_nstar)
        self.df_nstar_nested = self._add_Nmax(self.df_nstar_nested)

    @staticmethod
    def add_latex_column(df: pd.DataFrame, kernels: List = None) -> pd.DataFrame:
        """
        Copy of df with a column kernel_latex holding the LaTeX name of the kernel in column kernel. Kernels in
        `kernels` are matched by name; the others are parsed with Kernel.from_string.
        """
        known = {str(kernel_obj): kernel_obj for kernel_obj in (kernels or [])}
        latex = {name: (known[name] if name in known else Kernel.from_string(name)).latex_str()
                 for name in df["kernel"].unique()}
        out = df.copy()
        out["kernel_latex"] = out["kernel"].map(latex)
        return out

    def _add_latex_column_to_dfnstar(self):
        if self.df_nstar is not None:
            self.df_nstar = self.add_latex_column(self.df_nstar, self.kernels)
        if self.df_nstar_nested is not None:
            self.df_nstar_nested = self.add_latex_column(self.df_nstar_nested, self.kernels)

    def _load_preconfigured_plotting_parameters(self):
        sns.set(style="ticks", context="paper", font="times new roman")

        # mpl.use("TkAgg")
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = r"""
            \usepackage{mathptmx}
            \usepackage{amsmath}
        """
        font = {
            "family": "Times New Roman",
            "size": 10
        }
        mpl.rc("font", **font)

        # pretty names
        self.pretty_columns = {"alpha": r"$\alpha$", 'eps': r"$\varepsilon$", 'nstar': r"$n^*$",
                               'delta': r"$\delta$",
                               'N': r"$N$", 'nstar_absrel_error': "relative error", 'aq': r"$\varepsilon$",
                               'n': r"$n$",
                               "leq eps(delta)": rf"${self.validity_symbol}_{{n}}(P_{{N}},\varepsilon(\delta))$"}

        self.boxplot_args = dict(
            showfliers=False, palette="cubehelix",
            dodge=True, native_scale=False, fill=False, width=0.75, boxprops={"linewidth": 1.2}, gap=0.25
        )

        n_colors = self.dfmmd.nunique()["n"] if self.dfmmd is not None else 10
        self.lineplot_args = {
            "palette": sns.color_palette("crest_r", n_colors=n_colors, as_cmap=False),
        }

        self.axlines_args = {
            "lw": 1,
            "ls": "--",
            "color": "slategray"
        }

    @staticmethod
    def _plain_log_xticks(ax, labels: bool = True):
        """Plain decimal labels on a log x-axis, instead of the 3x10^-1 notation."""
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=(2, 3, 4, 6)))
        fmt = mpl.ticker.FuncFormatter(lambda x, _: rf"${x:g}$") if labels else mpl.ticker.NullFormatter()
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_formatter(fmt)

    def _get_df_nstar(self, resample: bool = True) -> pd.DataFrame:
        df = self.df_nstar if resample else self.df_nstar_nested
        if df is None:
            raise ValueError(f"No predicted nstar with resample={resample}: run "
                             f"ProjectManager.validity_analysis(resample={resample}) first.")
        return df

    def _compute_validity_df(self, deltas: List[float] = None) -> pd.DataFrame:
        """
        External validity (fraction of pairs of subsamples with MMD below eps(delta)), as a function of n.
        Only the largest N available for each configuration is used.
        """
        deltas = deltas if deltas is not None else self.config_params["delta"]

        if len(self.fixed_factors) == 0:
            Nmax = self.dfmmd["N"].max()
        else:
            Nmax = self.dfmmd.groupby(self.fixed_factors)["N"].transform("max")
        dfmmd = self.dfmmd.loc[self.dfmmd["N"] == Nmax]

        groupby_keys = self.fixed_factors + ["method", "n"]

        out = []
        for kernel_name, dfk in dfmmd.groupby("kernel"):
            kernel_obj = self._get_kernel(kernel_name)
            for delta in deltas:
                eps = kernel_obj.get_eps(delta, na=self.na)
                val = (dfk.assign(**{"leq eps(delta)": dfk["mmd"] <= eps})
                       .groupby(groupby_keys, as_index=False)["leq eps(delta)"].mean())
                out.append(val.assign(kernel=kernel_name, delta=delta, eps=eps))

        return self.add_latex_column(pd.concat(out, ignore_index=True), self.kernels)

    def plot_nstar_on_alpha_delta(self, alpha_fixed: float = 0.95, delta_fixed: float = 0.05, fig_width: float = 6.5,
                                  close_other_plots: bool = True, resample: bool = True, methods: List[str] = None,
                                  disjoint: bool = False, replace: bool = True):
        """
        Boxplots (over the configurations) of n* as a function of alpha (left, delta = delta_fixed) and of delta
        (right, alpha = alpha_fixed), for every kernel, at the largest N of every configuration.

        Parameters
        ----------
        alpha_fixed, delta_fixed : float
            The values of alpha and delta held fixed in the right and left panel.
        fig_width : float
            Width of the figure, in inches.
        close_other_plots : bool
            If True, close all open figures first.
        resample : bool
            Use the n* of the resampled (True) or nested (False) simulated studies.
        methods : list of str, optional
            Only use these n* estimation methods (e.g., ["embedding"]). Default: all methods.
        """

        if close_other_plots:
            plt.close("all")

        df_nstar = self._get_df_nstar(resample)
        if methods is not None:
            df_nstar = df_nstar.loc[df_nstar["method"].isin(methods)]
        df_nstar = df_nstar.query("")

        fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_width / 2), width_ratios=(1, 1), sharey=True)

        # ----  ALPHA
        ax = axes[0]
        dfplot = df_nstar.loc[(df_nstar["delta"] == delta_fixed) & (df_nstar["N"] == df_nstar["Nmax"])]

        # Make dfplot pretty
        dfplot = dfplot.rename(columns=self.pretty_columns)

        sns.boxplot(dfplot, x=self.pretty_columns["alpha"], y=self.pretty_columns["nstar"], ax=ax, hue="kernel_latex",
                    legend=False, **self.boxplot_args)
        ax.grid(color="grey", alpha=0.2)

        # ----  DELTA
        ax = axes[1]
        dfplot = df_nstar.loc[(df_nstar["alpha"] == alpha_fixed) & (df_nstar["N"] == df_nstar["Nmax"])]

        # Make dfplot pretty
        dfplot = dfplot.rename(columns=self.pretty_columns)

        sns.boxplot(dfplot, x=self.pretty_columns["delta"], y=self.pretty_columns["nstar"], ax=ax, hue="kernel_latex",
                    legend=True, **self.boxplot_args)
        ax.grid(color="grey", alpha=0.2)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend().remove()

        plt.tight_layout(pad=.5)
        plt.subplots_adjust(wspace=0.12, top=0.86)

        fig.legend(handles=handles, labels=labels, bbox_to_anchor=(0, 0.82 + 0.02, 1, 0.2),
                   loc="center", borderaxespad=1, ncol=dfplot.nunique()["kernel_latex"], frameon=False, fontsize=7)

        ax.set_yscale("log")

        sns.despine(right=True, top=True)
        if self.save:
            suffix = "" if resample else "_iterated"
            plt.savefig(self.figures_dir / f"{self.project_name}_nstar_alpha_delta{suffix}.pdf")
        if self.show:
            plt.show()

    def plot_validity_on_n(self, alpha: float = 0.95, deltas: List[float] = None, fig_width: float = 6.5,
                           aspect: float = 1.0, col_wrap: int = 2,
                           close_other_plots: bool = True, resample: bool = True):
        """
        External validity as a function of n, for every kernel (panels) and delta (hue): median and range over the
        configurations, at the largest N of every configuration. The dashed line marks alpha.

        Parameters
        ----------
        alpha : float
            Level marked by the horizontal line.
        deltas : list of float, optional
            Values of delta. Default: those in the configuration file.
        fig_width, aspect, col_wrap :
            Size and layout of the figure.
        close_other_plots : bool
            If True, close all open figures first.
        resample : bool
            Use the MMD of the resampled (True) or nested (False) simulated studies.
        """

        if close_other_plots:
            plt.close("all")

        dfmmd_backup = self.dfmmd
        self.set_resample(resample)
        try:
            dfplot = self._compute_validity_df(deltas).rename(columns=self.pretty_columns)
        finally:
            self.dfmmd = dfmmd_backup
        P = self.pretty_columns

        ncols = min(dfplot["kernel_latex"].nunique(), col_wrap)
        height = fig_width / (ncols * aspect)

        g = sns.relplot(
            data=dfplot, x=P["n"], y=P["leq eps(delta)"], hue=P["delta"], col="kernel_latex",
            kind="line", estimator="median", errorbar=("pi", 100),
            palette="flare_r", col_wrap=col_wrap, height=height, aspect=aspect,
        )
        g.set_titles(col_template="{col_name}")
        g.set(ylim=(0, 1.02))
        for ax in g.axes.flat:
            ax.axhline(alpha, **self.axlines_args)
        sns.move_legend(
            g, "lower center", bbox_to_anchor=(0.5, 1.0),
            ncol=dfplot[P["delta"]].nunique(), frameon=False,
        )

        plt.tight_layout(pad=.5)

        if self.save:
            suffix = "" if resample else "_iterated"
            g.savefig(self.figures_dir / f"{self.project_name}_validity_on_n{suffix}.pdf", bbox_inches="tight")
        if self.show:
            plt.show()

        return g

    def plot_simulated_experimental_study(self, configuration: dict, alpha: float, delta: float, fig_width: float = 6.5,
                                          xmin_padding: float = 0.8,
                                          close_other_plots: bool = True, kernels: List = None, resample: bool = True,
                                          method: str = None):
        """
        Simulated experimental studies of one configuration, one figure per kernel and one column per size N of the
        simulated study. Top: empirical CDF of the MMD (i.e., external validity as a function of epsilon) for every n.
        Bottom: alpha-quantiles of the MMD and the power-law fit log(n) = -2 log(q_alpha(n)) + b0, with the
        predicted n* at eps(delta).

        Parameters
        ----------
        configuration : dict
            Levels of the fixed factors, e.g., {"model": "DTC", "tuning": "no", "scoring": "F1"}.
        alpha, delta : float
            Desired probability and similarity threshold.
        fig_width : float
            Width of the figure, in inches.
        xmin_padding : float
            The x-axis starts at xmin_padding * eps(delta).
        close_other_plots : bool
            If True, close all open figures first.
        kernels : list of Kernel, optional
            Default: all kernels in the configuration file.
        resample : bool
            Use the MMD of the resampled (True) or nested (False) simulated studies.
        method : str, optional
            Only use the MMD estimated with this method. Default: all.
        """

        if close_other_plots:
            plt.close("all")

        self.set_resample(resample)

        kernels = kernels if kernels is not None else self.kernels
        for kernel_obj in kernels:

            eps = kernel_obj.get_eps(delta, na=self.na)

            xmin = eps * xmin_padding
            xmax = max(eps, self.dfmmd["mmd"].max())

            dfmmd_configuration = self.dfmmd.loc[configuration_mask(self.dfmmd, configuration)]
            if len(dfmmd_configuration.index) == 0:
                raise ValueError(f"Configuration {configuration} is not a valid configuration."
                                 f"To see the valid configurations: self.dfmmd.groupby(self.fixed_factors).groups")
            dfmmd_kernel = dfmmd_configuration.loc[dfmmd_configuration["kernel"] == str(kernel_obj)]
            if method is not None:
                dfmmd_kernel = dfmmd_kernel.loc[dfmmd_kernel["method"] == method]
            if len(dfmmd_kernel.index) == 0:
                raise ValueError(f"Kernel {kernel_obj} is not a valid kernel for configuration {configuration}."
                                 f"To see the valid kernels: "
                                 f"self.dfmmd.loc[configuration_mask(self.dfmmd, configuration), 'kernel'].unique()")

            Ns = np.sort(dfmmd_kernel["N"].unique())

            fig, axes = plt.subplots(3, len(Ns), figsize=(fig_width, 0.5 * fig_width), sharex=False, sharey="row",
                                     layout="constrained", height_ratios=[7, 7, 1], squeeze=False)

            kernel_latex = kernel_obj.latex_str().strip("$")
            for icol, Ncol in enumerate(Ns):

                dfplot = dfmmd_kernel.loc[dfmmd_kernel["N"] == Ncol]

                # alpha-quantile of the MMD for every n, as in ProjectManager._estimate_nstar_from_experiments
                dfaq = (dfplot.groupby("n")["mmd"].quantile(alpha, interpolation="higher")
                        .rename("aq").rename_axis("n").reset_index().rename(columns=self.pretty_columns))

                # -- external validity (MMD cdf)
                ax = axes[0, icol]

                ax.set_title(f"$N = {Ncol}$")
                ax.set_xlim(xmin, xmax)
                ax.set_xscale("log")

                ax.axhline(alpha, **self.axlines_args)
                ax.axvline(eps, **self.axlines_args)

                if icol == 0:
                    ax.set_ylabel(rf"${self.validity_symbol}^{{{kernel_latex}}}_n(P_N, \varepsilon)$")

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    sns.ecdfplot(dfplot, x="mmd", hue="n", ax=ax, legend=False, **self.lineplot_args)

                # Clean after seaborn
                ax.set_xlabel("")
                self._plain_log_xticks(ax, labels=False)

                # -- Linear regression
                ax = axes[1, icol]
                ax.set_xscale("log")
                ax.set_yscale("log")

                ax.axvline(eps, **self.axlines_args)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    sns.lineplot(dfaq, x=self.pretty_columns["aq"], y=self.pretty_columns["n"], ax=ax, ls="",
                                 marker="o",
                                 hue=self.pretty_columns["n"], legend=False, **self.lineplot_args)

                # Linear regression
                X = np.log(dfaq[self.pretty_columns["aq"]]).to_numpy().reshape(-1, 1)
                y = np.log(dfaq[self.pretty_columns["n"]]).to_numpy().reshape(-1, 1)
                epss = np.linspace(xmin, xmax, 1000)
                try:
                    # logn = -2 * logq + b0
                    b1 = -2
                    b0 = np.mean(y - b1 * X)

                    ns_pred = np.exp(b1 * np.log(epss) + b0)
                    nstar = int(np.ceil(np.exp(b1 * np.log(eps) + b0)))

                    ax.plot(epss, ns_pred, color="maroon", ls=":", alpha=0.7)
                    ax.plot(eps, nstar, marker='*', color='maroon', markersize=7)
                    ax.text(eps * 1.2, 1.2 * nstar, rf"$\hat{{n}}^*_{{{Ncol}}} = {nstar}$", color="maroon", fontsize=8)
                except (ValueError, OverflowError):
                    if self.verbose:
                        print(f"[WARNING] Failed linear regression for configuration: {dict2str(configuration)} and N: {Ncol}. Shape of X, y: {X.shape}, {y.shape}.")

                ax.set_xlim(xmin, xmax)
                self._plain_log_xticks(ax)

                # Turn off unnecessary axes (they're here to be replaced by the colormap)
                ax = axes[2, icol]
                ax.axis("off")

                # Clean after seaborn
                ax.set_xlabel("")
                ax.set_xticklabels([])

                # Add colormap
                if Ncol == max(Ns):
                    nmin, nmax = dfmmd_kernel["n"].min(), dfmmd_kernel["n"].max()
                    sm = plt.cm.ScalarMappable(cmap="crest_r", norm=plt.Normalize(nmin, nmax))
                    ax.figure.colorbar(sm, ax=axes[-1, :], location="bottom", shrink=0.5, extend="max", label="$n$",
                                       pad=0,
                                       fraction=1, ticks=range(int(nmin), int(nmax) + 1, 2))

            # - General formatting
            sns.despine(top=True, right=True)

            if self.save:
                plt.savefig(self.figures_dir / f"{self.project_name}_simulated_study__kernel={kernel_obj}.pdf")
            if self.show:
                plt.show()

    def plot_validity_sampling_comparison_on_n(self, alpha: float = 0.95, delta: float = 0.05,
                                                 fig_width: float = 6.5, aspect: float = 1.0, col_wrap: int = 2,
                                                 close_other_plots: bool = True):
        """
        External validity as a function of n for a fixed delta, estimated with resampled (resample=True) and iterated
        (resample=False) simulated studies. Same layout as plot_validity_on_n, with the hue on the sampling scheme.
        """

        if close_other_plots:
            plt.close("all")

        dfmmd_backup = self.dfmmd

        pretty_resample = {True: "fresh", False: "incremental"}

        dfrels = []
        try:
            for resample in (True, False):
                self.set_resample(resample)
                dfrels.append(self._compute_validity_df([delta]).assign(sampling=pretty_resample[resample]))
        finally:
            self.dfmmd = dfmmd_backup

        dfplot = pd.concat(dfrels, ignore_index=True).rename(columns=self.pretty_columns)
        P = self.pretty_columns

        ncols = min(dfplot["kernel_latex"].nunique(), col_wrap)
        height = fig_width / (ncols * aspect)

        g = sns.relplot(
            data=dfplot, x=P["n"], y=P["leq eps(delta)"], hue="sampling", col="kernel_latex",
            kind="line", estimator="median", errorbar=("pi", 100),
            palette="coolwarm", col_wrap=col_wrap, height=height, aspect=aspect,
        )
        g.set_titles(col_template="{col_name}")
        g.set(ylim=(0, 1.02))
        for ax in g.axes.flat:
            ax.axhline(alpha, **self.axlines_args)
        g.legend.set_title("")
        sns.move_legend(
            g, "lower center", bbox_to_anchor=(0.5, 1.0),
            ncol=dfplot["sampling"].nunique(), frameon=False,
        )

        plt.tight_layout(pad=.5)

        if self.save:
            g.savefig(self.figures_dir / f"{self.project_name}_validity_resampling_on_n__delta={delta}.pdf",
                      bbox_inches="tight")
        if self.show:
            plt.show()

        return g

    # def plot_nstar_method_comparison(self, ):
    #
    #     plt.close("all")
    #
    #     for kernel_obj in self.kernels:
    #         tmp = self.df_nstar.query("kernel == @kernel_obj.__str__()").drop(columns=["disjoint", "replace"])
    #         tmp_emb = tmp.query("method != 'approximation'").drop(columns="method")
    #         tmp_emb = tmp_emb.set_index([col for col in tmp_emb.columns if col != "nstar"])
    #         tmp_app = tmp.query("method == 'approximation'").drop(columns="method")
    #         tmp_app = tmp_app.set_index([col for col in tmp_app.columns if col != "nstar"])
    #
    #         dfplot = tmp_emb / tmp_app
    #         dfplot = dfplot.reset_index()
    #
    #         fig, axes = plt.subplots(1, 2)
    #         fig.suptitle(kernel_obj)
    #
    #         ax = axes[0]
    #         ax.set_title("Comparison embedding and approximation")
    #         sns.boxplot(data=dfplot, x="alpha", y="nstar", hue="delta", ax=ax, **self.boxplot_args)
    #         ax.axhline(1, color="grey", ls="--")
    #         ax.set_ylabel("emb/app")
    #
    #         ax = axes[1]
    #         ax.set_title("Median emb/app. Variation on the fixed configuration")
    #         dfplot2 = dfplot.groupby(["alpha", "delta"])["nstar"].agg(lambda x: np.median(np.abs(x))).reset_index()
    #         sns.scatterplot(data=dfplot2, x="alpha", y="delta", hue="nstar", palette="vlag", size="nstar", ax=ax)
    #
    #         # plt.get_current_fig_manager().window.state('zoomed')
    #         fig.show()
    #
    # def plot_nstar_approximation_comparison(self, configuration: dict, kernel_name: str = None,
    #                                         close_other_plots: bool = True):
    #
    #     query_str = self._get_query_string_from_configuration(dict(configuration, **{"N": self.dfmmd["N"].max(),
    #                                                                                  "kernel": kernel_name}))
    #
    #     mmd_cdf_symbol = r"$\hat F_n$"
    #     approx_cdf_symbol = r"$\sim \Phi_n$"
    #
    #     dfmmd1 = self.dfmmd.query(query_str)
    #     dfcoef1 = self.icdf_coefficients.query(query_str)
    #
    #     if close_other_plots:
    #         plt.close("all")
    #
    #     fig, ax = plt.subplots()
    #     for n in dfmmd1["n"].unique()[::2]:
    #         dfmmd2 = dfmmd1.query("n == @n")
    #
    #         xmin = np.quantile(dfmmd2["mmd"], 0.6)
    #         xmax = dfmmd2["mmd"].max()
    #
    #         L1, L4 = dfcoef1[["L1", "L4"]].values.flatten()
    #         # a = (L4 - L1 ** 2) / L1
    #         # k = 2 * L1 ** 2 / (L4 - L1 ** 2)
    #         # r = 1
    #
    #         # Y
    #         # Y = a * rng.chisquare(df=k, size=1000) ** r
    #
    #         # close formula
    #         epss = np.linspace(xmin, xmax, 1000)
    #         CF = self._mmd_cdf_approximation(epss, L1, L4, n)
    #
    #         sns.ecdfplot(data=dfmmd2, x="mmd", ax=ax, c="slategray", label=mmd_cdf_symbol)
    #         # sns.ecdfplot(x=np.sqrt(Y) / np.sqrt(n), ax=ax, label="Y", ls=":", c="orange")
    #         sns.lineplot(x=epss, y=CF, ax=ax, label=approx_cdf_symbol, ls="--", c="blue")
    #
    #     # fix legend
    #     h, l = ax.get_legend_handles_labels()
    #     h = [h[l.index(s)] for s in [mmd_cdf_symbol, approx_cdf_symbol]]
    #     l = [mmd_cdf_symbol, approx_cdf_symbol]
    #     ax.legend(h, l, frameon=False)
    #
    #     ax.set_xscale(r"log")
    #     ax.set_ylabel(r"\hat\Phi_n")
    #     ax.set_xlabel(r"$\varepsilon$")
    #     ax.set_xlim(10e-3, 2)
    #
    #     sns.despine(top=True, right=True)
    #
    #     fig.show()



if __name__ == "__main__":
    pm = ProjectManager(config_yaml_path="config.yaml", demo_dir=os.getcwd())
    df_nstar = pm.validity_analysis()
