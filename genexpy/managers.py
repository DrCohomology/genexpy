import ast
import os
import re
import shutil
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from functools import reduce
from itertools import product
from pathlib import Path
from typing import List, Literal, Union
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

from genexpy import kernels
from genexpy.kernels.base import Kernel
from genexpy.utils import rankings as ru
from genexpy import random as du


def dict2str(d: dict) -> str:
    s = str(d)
    s = re.sub(r"^\{(.*)}$", r"dict(\1)", s.strip())
    return re.sub(r"'(\w+)':", r"\1=", s)


def str2dict(s: str) -> dict:
    return ast.literal_eval(s)


class ProjectManager:
    """
    Manages the directories, loading the data from config and precomputed results, as well as

    """

    df_format = "parquet"

    def __init__(self, config_yaml_path: Union[str, Path], demo_dir: Union[str, Path], is_project_manager: bool = True):
        # --- Configuration and flags ---
        self.is_project_manager = is_project_manager
        self.demo_dir = Path(demo_dir)
        self.config_yaml_path = self.demo_dir / config_yaml_path
        self.project_name = None
        self.load_precomputed_mmd = None
        self.dump_results = None
        self.delete_existing_results = None

        # --- File and directory management ---
        self.figures_dir = None
        self.outputs_dir = None
        self.sample_mmd_dir = None  # contains resampled MMDs
        self.approx_mmd_dir = None  # contains CDF of the MMD

        # --- Core data structures ---
        self.estimation_methods = None
        self.dfmmd = None
        self.df_nstar = None
        self.df_nstar_nested = None
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
        )
        self.mmd_icdf_coefficients_filename_pattern = (
            r"configuration=dict(\(.*?\))__kernel=([A-Za-z0-9_]+\(.*?\))__N=(\d+)"
        )

        # --- Flags for experimental factors ---
        self.flag_design_factor = "_all"
        self.flag_held_constant_factor = None  # HC factors are those that are neither reliability nor design
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
            for resample in [True, False]:
                self._load_precomputed_mmd_df(resample=resample)
            if self.verbose:
                print("[INFO] Loaded existing results.")

    def _load_config_file(self):
        """Load experiment configuration from YAML and initialize project parameters."""

        config = self._read_yaml_config()
        self._initialize_paths(config["paths"])
        self._set_project_parameters(config["project_parameters"])
        self._validate_flags()

        self._prepare_config_data(config)
        config_params = self._prepare_config_params(config["parameters"])
        config_sampling = config["sampling"]
        config_methods = config["nstar_estimation_methods"]

        self._set_factor_lists()
        self.results = self._load_and_filter_dataset()
        self.config_data["factor_configurations"] = self._extract_factor_configurations()

        self.na = self.results.nunique()[self.config_data["alternatives_col_name"]]

        # Assign configuration to attributes
        self.config_params = config_params
        self.config_sampling = config_sampling
        self._load_kernels(config["kernels"], self.results)
        self.estimation_methods = config_methods

    # ---------------------- Subroutines ----------------------

    def _read_yaml_config(self) -> dict:
        with open(self.config_yaml_path, "r") as file:
            return yaml.safe_load(file)

    def _initialize_paths(self, paths_cfg: dict):
        try:
            if self.demo_dir != Path(os.getcwd()):
                os.chdir(self.demo_dir)
                print(f"[INFO] Moved working directory to {self.demo_dir}")
        except FileNotFoundError:
            print(f"[WARNING] Failed moving working directory to {self.demo_dir}.")

        self.outputs_dir = self.demo_dir / paths_cfg["outputs_dir"]
        self.figures_dir = self.demo_dir / "figures"
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
            if isinstance(params_cfg[key], float):
                params_cfg[key] = [params_cfg[key]]
        return params_cfg

    def _prepare_config_data(self, config: dict) -> None:
        assert sum(value is None for value in config["data"]["experimental_factors_name_lvl"].values()) == 1, (
            "Exactly one factor must be set to null in config.yaml."
        )
        self.config_data = config["data"]

    def _load_and_filter_dataset(self) -> pd.DataFrame:
        df = pd.read_parquet(self.config_data["dataset_path"])

        # Remove the columns not indicated as either factors, col of alternatives, or col of target (in config.yaml)
        tokeep = (self.all_factors + [self.config_data["alternatives_col_name"], self.config_data["target_col_name"]])
        df = df[tokeep]

        # Filter for the held-constant factors
        query_str = " and ".join(
            f"{factor} == '{lvl}'" if isinstance(lvl, str) else f"{factor} == {lvl}"
            for factor, lvl in self.config_data["experimental_factors_name_lvl"].items()
            if lvl not in [None, "_all"]
        )
        if len(query_str) == 0:
            return df
        return df.query(query_str).reset_index(drop=True)

    def _extract_factor_configurations(self) -> dict:
        try:
            return self.results.groupby(
                [
                    factor
                    for factor, lvl in self.config_data["experimental_factors_name_lvl"].items()
                    if lvl != self.flag_random_factor
                ]
            ).groups
        except ValueError:
            return {"None": self.results.index}

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

    def _load_kernels(self, kernels_cfg: list, df: pd.DataFrame):
        self.kernels = []
        for kernel_dict in kernels_cfg:
            kernel_params = kernel_dict["params"]

            param_combinations = product(
                *[[(param, val) for val in vals] for param, vals in kernel_params.items()]
            )

            for pc in param_combinations:
                alternatives = df[self.config_data["alternatives_col_name"]].unique()
                pc = dict(pc, **dict(na=len(alternatives), ordered_alternatives=alternatives))
                kernel_obj = Kernel.from_name_and_parameters(kernel_dict["name"], **pc)
                self.kernels.append(kernel_obj)

    def _create_project_directories(self):
        if self.delete_existing_results:
            shutil.rmtree(self.outputs_dir)

        self.sample_mmd_dir.mkdir(parents=True, exist_ok=True)
        readme = self.sample_mmd_dir / "README.md"
        readme.write_text("""
            This directory contains the precomputed distributions of the MMD. 
        """)

        self.approx_mmd_dir.mkdir(parents=True, exist_ok=True)
        readme = self.approx_mmd_dir / "README.md"
        readme.write_text("""
            This directory contains the coefficients for the ICDF (quantile function) of the MMD.
        """)

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        readme = self.figures_dir / "README.md"
        readme.write_text("""
            This directory contains the figures and plots. 
        """)

    def get_configurations(self, df_grouped: pd.DataFrame) -> dict:
        # Check that design and held-constant factors have been filtered correctly. They should have exactly one unique values
        if (df_grouped.nunique()[self.fixed_factors] > 1).any():
            raise ValueError("Factor levels not unique after query.")

        # Current levels of design and held-constant factor
        return dict(df_grouped[self.fixed_factors].iloc[0])

    def _get_configurations_and_grouped_df(self) -> list:
        df_grouped_list = [x[1] for x in self.results.groupby(self.fixed_factors)]
        configurations = [self.get_configurations(df_grouped) for df_grouped in df_grouped_list]

        return list(zip(configurations, df_grouped_list))

    @staticmethod
    def _get_query_string_from_configuration(configuration: dict):
        query_str = " and ".join(
            f"{factor} == '{lvl}'" if isinstance(lvl, str) else f"{factor} == {lvl}"
            for factor, lvl in configuration.items()
        )
        return query_str

    # ---- Routines to load and dump files
    def _load_preloaded_mmd_df(self, resample: bool = True):
        try:
            dfmmd = pd.read_parquet(self.outputs_dir / f"preloaded_mmd__resample={resample}.parquet")
        except FileNotFoundError:
            return False

        if not (dfmmd.get("resample", pd.Series(True, index=dfmmd.index)) == resample).all():
            warnings.warn(f"Preloaded MMD for resample={resample} does not match its flag. Reloading from files.")
            return False

        self.dfmmd = dfmmd
        return True

    def _load_precomputed_mmd_df(self, configuration_str: str = None, kernel_name: str = None, N: int = None,
                                 resample: bool = True, verbose: bool = False):

        if self._load_preloaded_mmd_df(resample):
            if verbose:
                print("[INFO] Loaded preloaded mmd dataframe.")
            return

        preloaded_path = self.outputs_dir / f"preloaded_mmd__resample={resample}.parquet"

        dfs = []
        for filepath in self.sample_mmd_dir.glob(f"*.{self.df_format}"):
            match self.df_format:
                case "parquet":
                    try:
                        matched_configuration, matched_kernel, matched_N, matched_resample = re.search(
                            self.precomputed_mmd_filename_pattern, str(filepath)).groups()
                    except AttributeError:
                        raise AttributeError(
                            f"File name {str(filepath)} is not in a valid pattern for the precomputed MMD files. "
                            f"The accepted patterns are {self.precomputed_mmd_filename_pattern}.")

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
                    if N is not None and matched_N != N:
                        continue

                    dfs.append(pd.read_parquet(filepath))

                case _:
                    raise NotImplementedError()
        if not dfs:
            self.dfmmd = None
            if verbose:
                print(f"[INFO] No precomputed MMD to load for resample={resample}.")
            return

        self.dfmmd = pd.concat(dfs, ignore_index=True)

        if verbose:
            print(f"[INFO] Loaded precomputed MMD for {len(self.precomputed_configurations)} configurations, "
                  f"{len(self.precomputed_kernels)} kernels, and {len(self.precomputed_Ns)} values of N.")

        self.dfmmd.to_parquet(preloaded_path)
        if verbose:
            print(f"[INFO] Dumped preloaded MMD dataframe in {preloaded_path}")

    def _load_preloaded_mmd_icdf(self):
        try:
            self.icdf_coefficiens = pd.read_parquet(self.outputs_dir / "preloaded_mmd_icdf_coeff.parquet")
            return True
        except FileNotFoundError:
            return False

    def _load_mmd_icdf_coefficients_df(self, configuration_str: str = None, kernel_name: str = None, N: int = None,
                                       verbose: bool = False):

        if self._load_preloaded_mmd_icdf():
            if verbose:
                print("[INFO] Loaded preloaded mmd icdf coefficients dataframe.")
            return

        dfs = []
        for filepath in self.approx_mmd_dir.glob(f"*.{self.df_format}"):
            match self.df_format:
                case "parquet":
                    try:
                        matched_configuration, matched_kernel, matched_N = re.search(
                            self.mmd_icdf_coefficients_filename_pattern, str(filepath)).groups()
                    except AttributeError:
                        raise AttributeError(
                            f"File name {str(filepath)} is not in a valid pattern for the precomputed MMD files. "
                            f"The accepted patterns are {self.precomputed_mmd_filename_pattern}.")

                    if configuration_str is not None and matched_configuration != configuration_str:
                        continue
                    if kernel_name is not None and matched_kernel != kernel_name:
                        continue
                    if N is not None and matched_N != N:
                        continue

                    dfs.append(pd.read_parquet(filepath))

                case _:
                    raise NotImplementedError()
        try:
            self.icdf_coefficiens = pd.concat(dfs, ignore_index=True)
        except ValueError:
            if verbose:
                print(f"No MMD ICDF coefficients to load.")

        if verbose:
            print(f"Loaded MMD ICDF coefficients.")

        self.icdf_coefficiens.to_parquet(self.outputs_dir / "preloaded_mmd_icdf_coeff.parquet")
        if verbose:
            print(f"[INFO] Dumped preloaded MMD dataframe in {self.outputs_dir / "preloaded_mmd_icdf_coeff.parquet"}")


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
        kernel_name = df.loc[0, "kernel"]
        N = df.loc[0, "N"]
        resample = df.loc[0, "resample"]
        configuration = dict2str(self.get_configurations(df))
        match self.df_format:
            case "parquet":
                df.to_parquet(
                    self.sample_mmd_dir / f"mmd__configuration={configuration}__kernel={kernel_name}"
                                          f"__N={N}__resample={resample}.parquet")
            case _:
                raise NotImplementedError()

    def _dump_mmd_icdf_coefficients_df(self, df: pd.DataFrame):
        kernel_name = df.loc[0, "kernel"]
        N = df.loc[0, "N"]
        configuration = dict2str(self.get_configurations(df))
        match self.df_format:
            case "parquet":
                df.to_parquet(
                    self.approx_mmd_dir / f"mmd_icdf__configuration={configuration}__kernel={kernel_name}__N={N}.parquet")
            case _:
                raise NotImplementedError()

    def _dump_nstar_df(self, resample: bool = True):
        if self.df_nstar is None:
            warnings.warn("No df_nstar to dump. Run reliability_analysis first to initialize it.")
            return

        match self.df_format:
            case "parquet":
                if self.df_nstar is not None:
                    self.df_nstar.to_parquet(self.outputs_dir / f"nstar_resample=True.{self.df_format}")
                if self.df_nstar_nested is not None:
                    self.df_nstar_nested.to_parquet(self.outputs_dir / f"nstar_resample=False.{self.df_format}")
            case _:
                raise NotImplementedError()

        if self.verbose:
            print(f"[INFO] Predicted nstar stored in {self.outputs_dir / f'nstar.{self.df_format}'}.")

    def _get_existing_precomputed_mmd(self, configuration: dict, kernel_obj: kernels.base.Kernel, N: int,
                                      resample: bool = True):
        if self.dfmmd is not None:
            precomputed_mmd = self.dfmmd.loc[self.dfmmd["kernel"] == str(kernel_obj)]
            precomputed_mmd = precomputed_mmd.query("N == @N")
            if "resample" in precomputed_mmd.columns:
                precomputed_mmd = precomputed_mmd.query("resample == @resample")
            elif not resample:
                return pd.DataFrame()  # legacy files without the flag are resampled
            for factor, lvl in configuration.items():
                precomputed_mmd = precomputed_mmd.query(f"{factor} == @lvl")
        else:
            precomputed_mmd = pd.DataFrame()

        return precomputed_mmd

    # ---- Routines to compute/estimate/approximate the MMD
    def _estimate_mmd_from_experiments_rankings(self, sample: ru.SampleAM, configuration: dict,
                                                kernel_obj: kernels.rankings.RankingKernel, N: int,
                                                method: Literal["naive", "vectorized", "embedding", "approximation"],
                                                resample: bool = True):

        # Get a subsample of size N
        if resample:
            distr = du.PMFDistribution.from_sample(sample)
            sample = distr.sample(N)

        precomputed_mmd = self._get_existing_precomputed_mmd(configuration, kernel_obj, N, resample)

        if not precomputed_mmd.empty:
            return precomputed_mmd
        else:
            dfmmd = kernel_obj.mmd_distribution_many_n(sample=sample, nmin=2, nmax=N // 2, step=2,
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

        # Get a subsample of size N
        if resample:
            rng = np.random.default_rng(seed=seed)
            s = rng.choice(s.T, size=N).T

        precomputed_mmd = self._get_existing_precomputed_mmd(configuration, kernel_obj, N, resample)

        if not precomputed_mmd.empty:
            return precomputed_mmd
        else:
            dfmmd = kernel_obj.mmd_distribution_many_n(s=s, nmin=2, nmax=N // 2, step=2,
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

    def estimate_mmd(self, sample, configuration, kernel_obj, N, method, resample: bool = True):

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
        Map the cdf of a variable distributed as a * chi_squared(df=k) with a normal.
        Parameters
        ----------
        y :
        k :
        a :

        Returns
        -------

        """
        return ((y / (a * k)) ** (1 / 3) - (1 - 2 / (9 * k))) / np.sqrt(2 / (9 * k))

    @staticmethod
    def wilson_hilferty_inv(z: np.ndarray, k: float, a: float) -> np.ndarray[float]:
        return a * k * (np.sqrt(2 / (9 * k)) * z + 1 - 2 / (9 * k)) ** 3

    @staticmethod
    def normal_cdf_Lin(x: np.ndarray):
        return 1 - 0.5 * np.exp(-0.717 * x - 0.416 * x ** 2)

    @staticmethod
    def normal_icdf_Lin(alpha: np.ndarray[float]) -> np.ndarray[float]:
        c0 = -0.861779
        c1 = 0.00120192
        c2 = 514089
        c3 = 1.664 * 10 ** 6
        return c0 + c1 * np.sqrt(c2 - c3 * np.log(-2 * (alpha - 1)))

    def _mmd_cdf_approximation(self, eps: np.ndarray, L1: float, L4: float, n: int) -> float:
        """
        Evaluate the approximated CDF of the MMD.
        Close-formula approximation via the folliwng steps:
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

        a = (L4 - L1 ** 2) / L1
        k = 2 * L1 ** 2 / (L4 - L1 ** 2)

        return np.sqrt(self.wilson_hilferty_inv(self.normal_icdf_Lin(alpha), k, a) / n)

    def _get_nstar_mmd_icdf_approximation(self, eps: float, alpha: float, L1: float, L4: float) -> float:
        """
        Evaluate the approximation of the inverse cumulative function (ICDF) of the MMD.

        Parameters
        ----------
        eps :

        Returns
        -------

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
                                           kernel_obj: kernels.base.Kernel, N: int = None, resample:bool = True):
        """
        use the approximation of the MMD using the approximate CDF of a normal variable. Details
            in the paper Matteucci et al. (2025).

        Returns
        -------

        Literature
        ----------
        Matteucci et al. (2025): Matteucci, Federico, et al. "reliability of experimental studies." arXiv preprint arXiv:2406.17374 (2024).
        Lin (1989): Lin, Jinn‐Tyan. "Approximating the normal tail probability and its inverse for use on a pocket calculator." Journal of the Royal Statistical Society: Series C (Applied Statistics) 38.1 (1989): 69-70.
        Choudhury (2007) : Choudhury, Amit, Subhasis Ray, and Pradipta Sarkar. "Approximating the cumulative distribution function of the normal distribution." Journal of Statistical Research 41.1 (2007): 59-67.
        """

        alphas = np.array(self.config_params["alpha"])
        deltas = np.array(self.config_params["delta"])

        if np.min(alphas) < 0.6:
            warnings.warn("The approximation of the MMD might not be reliable for alpha < 0.6.")

        if isinstance(sample, ru.SampleAM) and isinstance(kernel_obj, kernels.rankings.RankingKernel):
            if N is not None and resample:
                distr = du.PMFDistribution.from_sample(sample)
                sample = distr.sample(N)

            support, pmf = sample.get_support_pmf()
            x = kernel_obj._convert_sample_to_input_format(support)
            K = kernel_obj.gram_matrix(x, x)
            m = len(support)

        elif isinstance(sample, np.ndarray) and isinstance(kernel_obj, kernels.vectors.VectorKernel):
            if N is not None and resample:
                rng = np.random.default_rng(seed=4637843)
                sample = rng.choice(sample.T, size=N).T

            support, counts = np.unique(tuple(tuple(x) for x in sample), axis=1, return_counts=True)
            pmf = counts / np.sum(counts)
            K = kernel_obj.gram_matrix(support, support)
            m = support.shape[1]

        else:
            raise TypeError(f"Parameter sample of type {type(sample)} is not a valid input type.")

        C = np.eye(m) - 1 / m * np.ones((m, m))
        H = C @ K @ C

        Th = H @ np.diag(pmf)

        lam = np.linalg.eigvalsh(Th)

        L1 = np.sum(lam)
        L2 = np.sum(lam ** 2)
        L3 = np.sum(np.triu(np.outer(lam, lam), 1))
        L4 = 3 * L2 + 2 * L3

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

        coefficients = dict(configuration, **dict(kernel=str(kernel_obj), N=N, L1=L1, L4=L4))
        self._dump_mmd_icdf_coefficients_df(pd.DataFrame(coefficients, index=[0]))

        return out

    def estimate_nstar(self, sample: Union[ru.SampleAM, np.ndarray[float]], configuration: dict,
                       kernel_obj: kernels.base.Kernel, *args,
                       method: Literal["naive", "vectorized", "embedding", "approximation"] = "embedding",
                       out: List = None, **kwargs):

        out = out if out is not None else []
        if method == "approximation":
            tmp = self._estimate_nstar_from_approximation(sample, configuration, kernel_obj, *args, **kwargs)
        elif method in ["naive", "vectorized", "embedding"]:
            tmp = self._estimate_nstar_from_experiments(sample, configuration, kernel_obj, method=method, **kwargs)
        else:
            raise ValueError(f"Parameter method={method} is not a valid option. "
                             f"Valid options are 'naive', 'vectorized', 'embedding', 'approximation'.")

        return out.extend(tmp) or out

    def _reliability_analysis_one_configuration(self, sample_rankings: ru.SampleAM,
                                                     sample_vectors: np.ndarray[float], out: List = None,
                                                     configuration: dict = None):

        configuration = configuration if configuration is not None else dict()
        out = out if out is not None else []

        # Loop over the kernels
        for kernel_obj in self.kernels:
            # Set the Universe
            kernel_obj.set_support(sample_rankings.get_support_pmf()[0])

            for N in range(self.config_sampling["sample_size"], np.nanmin((len(sample_rankings),
                                                                          self.config_params["Nmax"])),
                           self.config_sampling["sample_size"]):
                if isinstance(kernel_obj, kernels.rankings.RankingKernel):
                    for method in self.estimation_methods["rankings"]:
                        out = self.estimate_nstar(sample=sample_rankings, configuration=configuration,
                                                  kernel_obj=kernel_obj, method=method, out=out, N=N)
                elif isinstance(kernel_obj, kernels.vectors.VectorKernel):
                    for method in self.estimation_methods["vectors"]:
                        out = self.estimate_nstar(sample=sample_vectors, configuration=configuration,
                                                  kernel_obj=kernel_obj, method=method, out=out, N=N)
                else:
                    raise TypeError(f"Parameter kernel_obj with type {type(kernel_obj)} is invalid. Valid inputs are "
                                    f"kernels.vector.VectorKernel or kernels.rankings.RankingKernel")

        return out

    def _reliability_analysis_one_configuration_nested(self, sample_rankings: ru.SampleAM,
                                                       sample_vectors: np.ndarray[float], out: List = None,
                                                       configuration: dict = None, seed: int = None):
        """
        As _reliability_analysis_one_configuration, but with nested subsamples: the subsample of size N + Nstep
        keeps the N experiments already drawn and adds Nstep new ones, without replacement.
        """

        configuration = configuration if configuration is not None else dict()
        out = out if out is not None else []

        Nstep = self.config_sampling["sample_size"]
        Nmax = int(np.nanmin((len(sample_rankings), self.config_params["Nmax"])))
        order = np.random.default_rng(seed=seed).permutation(len(sample_rankings))

        # Loop over the kernels
        for kernel_obj in self.kernels:
            # Set the Universe
            kernel_obj.set_support(sample_rankings.get_support_pmf()[0])

            for N in range(Nstep, Nmax, Nstep):
                idx = order[:N]
                if isinstance(kernel_obj, kernels.rankings.RankingKernel):
                    for method in self.estimation_methods["rankings"]:
                        out = self.estimate_nstar(sample=ru.SampleAM(np.asarray(sample_rankings)[idx]),
                                                  configuration=configuration,
                                                  kernel_obj=kernel_obj, method=method, out=out, N=N, resample=False)
                elif isinstance(kernel_obj, kernels.vectors.VectorKernel):
                    for method in self.estimation_methods["vectors"]:
                        out = self.estimate_nstar(sample=sample_vectors[:, idx], configuration=configuration,
                                                  kernel_obj=kernel_obj, method=method, out=out, N=N, resample=False)
                else:
                    raise TypeError(f"Parameter kernel_obj with type {type(kernel_obj)} is invalid. Valid inputs are "
                                    f"kernels.vector.VectorKernel or kernels.rankings.RankingKernel")

        return out
#
    def reliability_analysis(self, resample: bool = True):
        """

        Parameters
        ----------
        resample : if True, samples are redrawn entirely. If False, samples are built iteratively

        Returns
        -------

        """


        self.results_rankings = ru.get_matrix_from_df(self.results, factors=list(self.all_factors),
                                                      alternatives=self.config_data["alternatives_col_name"],
                                                      target=self.config_data["target_col_name"],
                                                      get_rankings=True,
                                                      lower_is_better=self.config_data["target_is_error"],
                                                      impute_missing=True,
                                                      tol_missing_indices=self.config_params[
                                                          "tol_missing_alternatives"],
                                                      tol_missing_columns=self.config_params["tol_missing_conditions"],
                                                      as_numpy=False)

        self.results_matrix = ru.get_matrix_from_df(self.results, factors=list(self.all_factors),
                                                    alternatives=self.config_data["alternatives_col_name"],
                                                    target=self.config_data["target_col_name"],
                                                    get_rankings=False, impute_missing=True,
                                                    tol_missing_indices=self.config_params["tol_missing_alternatives"],
                                                    tol_missing_columns=self.config_params["tol_missing_conditions"],
                                                    as_numpy=False)

        if self.verbose:
            na_tmp = self.results.nunique()[self.config_data['alternatives_col_name']]
            print(f"[INFO] Kept {self.results_rankings.shape[0]} / {na_tmp} indices (alternatives) and "
                  f"{self.results_rankings.shape[1]} / {len(self.results.groupby(self.all_factors))} columns (conditions).")
            print(f"[INFO] Starting the reliability analysis.")

        if self.verbose:
            iterator = tqdm(self._get_configurations_and_grouped_df(),
                            position=0, desc="Configurations", leave=True)
        else:
            iterator = self._get_configurations_and_grouped_df()

        out = []
        for configuration, _ in iterator:
            mask = pd.Series(True, index=self.results_rankings.columns)
            for col_level, value in configuration.items():
                mask &= (self.results_rankings.columns.get_level_values(col_level) == value)

            rankings = self.results_rankings.loc[:, mask.values]
            sample_rankings = ru.SampleAM.from_rank_vector_matrix(rankings.values)
            sample_vectors = self.results_matrix.loc[:, mask.values].values

            if resample:
                out = self._reliability_analysis_one_configuration(sample_rankings=sample_rankings,
                                                                        sample_vectors=sample_vectors, out=out,
                                                                        configuration=configuration)
            else:
                out = self._reliability_analysis_one_configuration_nested(sample_rankings=sample_rankings,
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

        return self.df_nstar




class PlotManager(ProjectManager):
    def __init__(self, config_yaml_path: Union[str, Path], demo_dir: Union[str, Path], save: bool = True,
                 show: bool = True):
        super().__init__(config_yaml_path, is_project_manager=False, demo_dir=demo_dir)

        self.show = show
        self.save = save
        self.boxplot_args = None
        self.pretty_kernels = None
        self.pretty_columns = None
        self.df_nstar = None
        self.df_nstar_nested = None
        self.dfmmd = None

        self._load_nstar_df()
        self._add_Nmax_column_to_dfnstar()
        self._add_latex_column_to_dfnstar()
        self.dfmmds = {}
        for resample in (True, False):
            self.dfmmd = None
            self._load_precomputed_mmd_df(resample=resample)
            self.dfmmds[resample] = self.dfmmd
        self.set_resample()
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

    def _add_Nmax_column_to_dfnstar(self):
        self.df_nstar = self.df_nstar.join(self.df_nstar.groupby(self.fixed_factors)["N"].max(),
                                           on=self.fixed_factors, rsuffix="max")

    @staticmethod
    def add_latex_column(df):
        out = df.copy()
        df["kernel_latex"] = df["kernel"].apply(lambda x: Kernel.from_string(x).latex_str())
        return df

    def _add_latex_column_to_dfnstar(self):
       self.df_nstar = self.add_latex_column(self.df_nstar)


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
                               "leq eps(delta)": r"$\text{R}^k_{n}(P_{N},\varepsilon(\delta))$"}  # columns

        # self.pretty_kernels = {"borda_kernel_idx_OHE": r"$\kappa_b^{\text{OHE}, 1/n}$",
        #                        "mallows_kernel_nu_auto": r"$\kappa_m^{1/\binom{n}{2}}$",
        #                        "jaccard_kernel_k_1": r"$\kappa_j^{1}$"}  # kernels
        # self.pretty_kernels.update({"borda_kernel_idx_OHE": "$g_1$", "mallows_kernel_nu_auto": "$g_3$",
        #                             "jaccard_kernel_k_1": "$g_2$"})  # rename to goal_1, 2, 3

        self.boxplot_args = dict(
            showfliers=False, palette="cubehelix",
            dodge=True, native_scale=False, fill=False, width=0.75, boxprops={"linewidth": 1.2}, gap=0.25
        )

        self.lineplot_args = {
            "palette": sns.color_palette("crest_r", n_colors=self.dfmmd.nunique()["n"], as_cmap=False),
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

    def _compute_reliability_df(self, deltas: List[float] = None) -> pd.DataFrame:
        """
        Fraction of resampled experiments with MMD below eps(delta), as a function of n.
        Only the largest N available for each configuration is used.
        """
        deltas = deltas if deltas is not None else self.config_params["delta"]

        dfmmd = self.dfmmd.join(self.dfmmd.groupby(self.fixed_factors)["N"].max(),
                                on=self.fixed_factors, rsuffix="max").query("N == Nmax")

        groupby_keys = self.fixed_factors + ["method", "n"]

        out = []
        for kernel_name, dfk in dfmmd.groupby("kernel"):
            kernel_obj = Kernel.from_string(kernel_name)
            for delta in deltas:
                eps = kernel_obj.get_eps(delta, na=self.na)
                rel = (dfk.assign(**{"leq eps(delta)": dfk["mmd"] <= eps})
                       .groupby(groupby_keys, as_index=False)["leq eps(delta)"].mean())
                out.append(rel.assign(kernel=kernel_name, delta=delta, eps=eps))

        return self.add_latex_column(pd.concat(out, ignore_index=True))

    def plot_nstar_on_alpha_delta(self, alpha_fixed: float = 0.95, delta_fixed: float = 0.05, fig_width: float = 6.5,
                                  close_other_plots: bool = True, resample: bool = True):

        if close_other_plots:
            plt.close("all")

        if not resample:
            raise NotImplementedError("resample=False is not implemented for this method.")

        fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_width / 2), width_ratios=(1, 1), sharey=True)

        # ----  ALPHA
        ax = axes[0]
        dfplot = self.df_nstar.loc[
            (self.df_nstar["delta"] == delta_fixed) & (self.df_nstar["N"] == self.df_nstar["Nmax"])]

        # Make dfplot pretty
        dfplot = dfplot.rename(columns=self.pretty_columns)

        sns.boxplot(dfplot, x=self.pretty_columns["alpha"], y=self.pretty_columns["nstar"], ax=ax, hue="kernel_latex",
                    legend=False, **self.boxplot_args)
        ax.grid(color="grey", alpha=0.2)

        # ----  DELTA
        ax = axes[1]
        dfplot = self.df_nstar.loc[
            (self.df_nstar["alpha"] == alpha_fixed) & (self.df_nstar["N"] == self.df_nstar["Nmax"])]

        # Make dfplot pretty
        dfplot = dfplot.rename(columns=self.pretty_columns)

        sns.boxplot(dfplot, x=self.pretty_columns["delta"], y=self.pretty_columns["nstar"], ax=ax, hue="kernel_latex",
                    legend=True, **self.boxplot_args)
        ax.grid(color="grey", alpha=0.2)

        # ax.legend(*ax.get_legend_handles_labels()).get_frame().set_edgecolor("w")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend().remove()

        plt.tight_layout(pad=.5)
        plt.subplots_adjust(wspace=0.12, top=0.86)

        fig.legend(handles=handles, labels=labels, bbox_to_anchor=(0, 0.82 + 0.02, 1, 0.2),
                   loc="center", borderaxespad=1, ncol=dfplot.nunique()["kernel_latex"], frameon=False)

        ax.set_yscale("log")

        sns.despine(right=True, top=True)
        if self.save:
            plt.savefig(self.figures_dir / f"{self.project_name}_nstar_alpha_delta.pdf")
        if self.show:
            plt.show()

    def plot_reliability_on_n(self, alpha: float = 0.95, deltas: List[float] = None, fig_width: float = 6.5,
                              aspect: float = 1.0, col_wrap: int = 2,
                              close_other_plots: bool = True, resample: bool = True):

        if not resample:
            raise NotImplementedError("resample=False is not implemented for this method.")

        if close_other_plots:
            plt.close("all")

        dfplot = self._compute_reliability_df(deltas).rename(columns=self.pretty_columns)
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
            g.savefig(self.figures_dir / f"{self.project_name}_reliability_on_nstar.pdf", bbox_inches="tight")
        if self.show:
            plt.show()

        return g

    def plot_simulated_experimental_study(self, configuration: dict, alpha: float, delta: float, fig_width: float = 6.5,
                                          xmin_padding: float = 0.8,
                                          close_other_plots: bool = True, kernels: List = None, resample: bool = True):
        """
        Predictions for nstar based on the non-approximating methods.
        Loads the precomputed MMD files.

        Parameters
        ----------
        close_other_plots :
        fig_width :
        alpha :
        delta :

        Returns
        -------

        """

        if close_other_plots:
            plt.close("all")

        self.set_resample(resample)

        kernels = kernels if kernels is not None else self.kernels
        for kernel_obj in kernels:

            eps = kernel_obj.get_eps(delta, na=self.na)

            xmin = eps * xmin_padding
            xmax = max(eps, self.dfmmd["mmd"].max()) #/ padding

            query_str = self._get_query_string_from_configuration(configuration)
            dfmmd_configuration = self.dfmmd.query(query_str)
            if len(dfmmd_configuration.index) == 0:
                raise ValueError(f"Configuration {configuration} is not a valid configuration."
                                 f"To see the valid configurations: self.dfmmd.groupby(self.configuration_factors).groups")
            dfmmd_kernel = dfmmd_configuration.query("kernel == @kernel_obj.__str__()")
            if len(dfmmd_kernel.index) == 0:
                raise ValueError(f"Kernel {kernel_obj} is not a valid kernel for configuration {configuration}."
                                 f"To see the valid kernels: self.dfmmd.query(self._get_query_string_from_configuration(configuration))['kernels'].unique()")

            Ns = dfmmd_kernel["N"].unique()

            fig, axes = plt.subplots(3, len(Ns), figsize=(fig_width, 0.7 * fig_width), sharex=False, sharey="row",
                                     layout="constrained", height_ratios=[7, 7, 1])

            for icol, Ncol in enumerate(Ns):

                dfplot = dfmmd_kernel.query("N == @Ncol")

                dfq = (dfplot.groupby("n")["mmd"].quantile(self.config_params["alpha"], interpolation="higher")
                       .rename("q_alpha").rename_axis(index=["n", "alpha"]).reset_index())

                alpha_quantiles = {
                    n: np.quantile(np.quantile(dfq.query("n == @n")["q_alpha"], alpha, method="linear"), alpha)
                    for n in dfq["n"].unique()}
                dfaq = pd.DataFrame(alpha_quantiles, index=[0]).melt(var_name="n", value_name="aq").rename(
                    columns=self.pretty_columns)

                # -- reliability (MMD cdf)
                ax = axes[0, icol]

                ax.set_title(f"$N = {Ncol}$")
                ax.set_xlim(xmin, xmax)
                ax.set_xscale("log")

                ax.axhline(alpha, **self.axlines_args)
                ax.axvline(eps, **self.axlines_args)

                if icol == 0:
                    ax.set_ylabel(fr"$\text{{R}}^{{{kernel_obj.latex_str().strip("$")}}}_n(P_N, \varepsilon)$")

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    sns.ecdfplot(dfplot, x="mmd", hue="n", ax=ax, legend=False, **self.lineplot_args)

                # Clean after seaborn
                ax.set_xlabel("")
                self._plain_log_xticks(ax, labels=False)

                # Quantile lines
                # for (n, laq), color in zip(alpha_quantiles.items(), sns.color_palette(palette, n_colors=len(alpha_quantiles))):
                #     ax.vlines(laq, ymin=0, ymax=alpha, ls="-", color=color, lw=lw)
                # ax.axvline(laq, ymin=-1.2, ymax=0, ls=":", color=color, lw=lw, zorder=-1, clip_on=False)
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
                    nstar = int(np.exp(b1 * np.log(eps) + b0))

                    # lr = LinearRegression()
                    # lr.fit(X, y)
                    # ns_pred = np.exp(lr.predict(np.log(epss).reshape(-1, 1)).reshape(1, -1)[0])
                    # nstar = int(ns_pred[np.argmin(np.abs(epss - eps))])

                    ax.plot(epss, ns_pred, color="maroon", ls=":", alpha=0.7)
                    ax.plot(eps, nstar, marker='*', color='maroon', markersize=7)
                    ax.text(eps * 1.2, 1.2 * nstar, rf"$\hat{{n}}^*_{{{Ncol}}} = {nstar}$", color="maroon", fontsize=8)
                except ValueError:
                    if self.verbose:
                        print(f"[WARNING] Failed linear regression for configuration: {dict2str(configuration)} and N: {Ncol}. Shape of X, y: {X.shape}, {y.shape}.")

                # Quantile lines
                # for (n, aq), color in zip(alpha_quantiles.items(), sns.color_palette(palette, n_colors=len(alpha_quantiles))):
                #     ax.vlines(aq, ymin=n, ymax=nstar, ls="-", color=color, lw=lw)
                # ax.axvline(aq, ymin=1, ymax=1.2, ls=":", color=color, lw=lw, zorder=-1, clip_on=False)
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

    def plot_reliability_resampling_comparison_on_n(self, alpha: float = 0.95, delta: float = 0.05,
                                                    fig_width: float = 6.5, aspect: float = 1.0, col_wrap: int = 2,
                                                    close_other_plots: bool = True):
        """
        Reliability as a function of n for a fixed delta, estimated with resample=True and with resample=False.
        Same layout as plot_reliability_on_n, with the hue on the sampling scheme.
        """

        if close_other_plots:
            plt.close("all")

        dfmmd_backup = self.dfmmd

        pretty_resample = {True: "resampled", False: "nested"}

        dfrels = []
        for resample in (True, False):
            self.set_resample(resample)
            dfrels.append(self._compute_reliability_df([delta]).assign(sampling=pretty_resample[resample]))
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
            g.savefig(self.figures_dir / f"{self.project_name}_reliability_resampling_on_nstar__delta={delta}.pdf",
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
    #     dfcoef1 = self.icdf_coefficiens.query(query_str)
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


if __name__ == "__main__()":
    import os

    pm = ProjectManager(config_yaml_path="config.yaml", demo_dir=os.getcwd())
    df_nstar = pm.reliability_analysis()
