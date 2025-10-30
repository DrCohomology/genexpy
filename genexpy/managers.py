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

    def __init__(self, config_yaml_path: Union[str, Path], is_project_manager: bool = True):
        # --- Configuration and flags ---
        self.is_project_manager = is_project_manager
        self.config_yaml_path = Path(config_yaml_path)
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
        self.factors_dict = None
        self.results = None
        self.kernels = None

        # --- Factors ---
        self.all_factors: list = []
        self.design_factors: list = []
        self.generalizability_factors: list = []
        self.held_constant_factors: list = []
        self.configuration_factors: list = []  # design + held-constant factors

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
            r"configuration=dict(\(.*?\))__kernel=([A-Za-z0-9_]+\(.*?\))__N=(\d+)"
        )
        self.mmd_icdf_coefficients_filename_pattern = (
            r"configuration=dict(\(.*?\))__kernel=([A-Za-z0-9_]+\(.*?\))__N=(\d+)"
        )

        # --- Flags for experimental factors ---
        self.flag_design_factor = "_all"
        self.flag_held_constant_factor = None  # HC factors are those that are neither generalizability nor design
        self.flag_generalizability_factor = None

        # --- Initialization steps ---
        self._move_to_script_directory()
        self._load_config_file()
        if self.is_project_manager:
            self._create_project_directories()

        if self.load_precomputed_mmd:
            self._load_precomputed_mmd_df()

    @staticmethod
    def _move_to_script_directory():
        try:
            os.chdir("demos/Template")
        except FileNotFoundError:
            pass

    def _load_config_file(self):
        """Load experiment configuration from YAML and initialize project parameters."""

        config = self._read_yaml_config()
        self._initialize_paths(config["paths"])
        self._set_project_parameters(config["project_parameters"])
        self._validate_flags()

        config_data = self._prepare_config_data(config["data"])
        config_params = self._prepare_config_params(config["parameters"])
        config_sampling = config["sampling"]
        config_methods = config["nstar_estimation_methods"]

        df = self._load_and_filter_dataset(config_data)
        config_data["factor_configurations"] = self._extract_factor_configurations(df, config_data)
        config_data["results"] = df

        self._set_factor_lists(config_data)
        self.na = df.nunique()[config_data["alternatives_col_name"]]

        # Assign configuration to attributes
        self.config_data = config_data
        self.config_params = config_params
        self.config_sampling = config_sampling
        self._load_kernels(config["kernels"], df)
        self.estimation_methods = config_methods
        self.results = config_data["results"]

    # ---------------------- Subroutines ----------------------

    def _read_yaml_config(self) -> dict:
        with open(self.config_yaml_path, "r") as file:
            return yaml.safe_load(file)

    def _initialize_paths(self, paths_cfg: dict):
        self.outputs_dir = Path(paths_cfg["outputs_dir"])
        self.figures_dir = self.outputs_dir / "figures"
        self.sample_mmd_dir = self.outputs_dir / "MMD_precomputed"
        self.approx_mmd_dir = self.outputs_dir / "MMD_approximated_icdf_coefficients"

    def _set_project_parameters(self, general_cfg: dict):
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

    def _prepare_config_data(self, data_cfg: dict) -> dict:
        assert sum(value is None for value in data_cfg["experimental_factors_name_lvl"].values()) == 1, (
            "Exactly one factor must be set to null in config.yaml."
        )
        return data_cfg

    def _load_and_filter_dataset(self, config_data: dict) -> pd.DataFrame:
        df = pd.read_parquet(config_data["dataset_path"])
        query_str = " and ".join(
            f"{factor} == '{lvl}'" if isinstance(lvl, str) else f"{factor} == {lvl}"
            for factor, lvl in config_data["experimental_factors_name_lvl"].items()
            if lvl not in [None, "_all"]
        )
        if len(query_str) == 0:
            return df

        return df.query(query_str).reset_index(drop=True)

    def _extract_factor_configurations(self, df: pd.DataFrame, config_data: dict) -> dict:
        try:
            return df.groupby(
                [
                    factor
                    for factor, lvl in config_data["experimental_factors_name_lvl"].items()
                    if lvl != self.flag_generalizability_factor
                ]
            ).groups
        except ValueError:
            return {"None": df.index}

    def _set_factor_lists(self, config_data: dict):
        self.all_factors = list(config_data["experimental_factors_name_lvl"].keys())
        self.design_factors = [
            f for f, lvl in config_data["experimental_factors_name_lvl"].items()
            if lvl == self.flag_design_factor
        ]
        self.generalizability_factors = [
            f for f, lvl in config_data["experimental_factors_name_lvl"].items()
            if lvl == self.flag_generalizability_factor
        ]
        self.held_constant_factors = [
            f for f, lvl in config_data["experimental_factors_name_lvl"].items()
            if lvl not in [self.flag_design_factor, self.flag_generalizability_factor]
        ]
        self.configuration_factors = self.design_factors + self.held_constant_factors

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
            print(f"⚠️ Removed existing directory: {self.outputs_dir}.")

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

        f"📁 Created new output directory: {self.outputs_dir}."

    def get_configurations(self, df_grouped: pd.DataFrame) -> dict:
        # Check that design and held-constant factors have been filtered correctly. They should have exactly one unique values
        if (df_grouped.nunique()[self.configuration_factors] > 1).any():
            raise ValueError("Factor levels not unique after query.")

        # Current levels of design and held-constant factor
        return dict(df_grouped[self.configuration_factors].iloc[0])

    def _get_configurations_and_grouped_df(self) -> list:
        df_grouped_list = [x[1] for x in self.results.groupby(self.configuration_factors)]
        configurations = [self.get_configurations(df_grouped) for df_grouped in df_grouped_list]

        return [(configuration, df_grouped) for configuration, df_grouped in zip(configurations, df_grouped_list)]

    @staticmethod
    def _get_query_string_from_configuration(configuration: dict):
        query_str = " and ".join(
            f"{factor} == '{lvl}'" if isinstance(lvl, str) else f"{factor} == {lvl}"
            for factor, lvl in configuration.items()
        )
        return query_str

    # ---- Routines to load and dump files
    def _load_precomputed_mmd_df(self, configuration_str: str = None, kernel_name: str = None, N: int = None,
                                 verbose: bool = False):
        dfs = []
        for filepath in self.sample_mmd_dir.glob(f"*.{self.df_format}"):
            match self.df_format:
                case "parquet":
                    try:
                        matched_configuration, matched_kernel, matched_N = re.search(
                            self.precomputed_mmd_filename_pattern, str(filepath)).groups()
                    except AttributeError:
                        raise AttributeError(
                            f"File name {str(filepath)} is not in a valid pattern for the precomputed MMD files. "
                            f"The accepted patterns are {self.precomputed_mmd_filename_pattern}.")

                    self.precomputed_configurations.add(matched_configuration)
                    self.precomputed_kernels.add(matched_kernel)
                    self.precomputed_Ns.add(matched_N)

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
            self.dfmmd = pd.concat(dfs, ignore_index=True)
        except ValueError:
            if verbose:
                print(f"No precomputed MMD to load.")

        if verbose:
            print(
                f"Loaded precomputed MMD for {len(self.precomputed_configurations)} configurations, {len(self.precomputed_kernels)} kernels, and {len(self.precomputed_Ns)} values of N.")


    def _load_mmd_icdf_coefficients_df(self, configuration_str: str = None, kernel_name: str = None, N: int = None,
                                 verbose: bool = False):
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
            print(
                f"Loaded MMD ICDF coefficients.")

    def _load_nstar_df(self, force=True):
        if self.df_nstar is not None and not force:
            return

        match self.df_format:
            case "parquet":
                self.df_nstar = pd.read_parquet(self.outputs_dir / f"nstar.{self.df_format}")
            case _:
                raise NotImplementedError()

    def _dump_sample_mmd_df(self, df: pd.DataFrame):
        kernel_name = df.loc[0, "kernel"]
        N = df.loc[0, "N"]
        configuration = dict2str(self.get_configurations(df))
        match self.df_format:
            case "parquet":
                df.to_parquet(self.sample_mmd_dir / f"mmd__configuration={configuration}__kernel={kernel_name}__N={N}.parquet")
            case _:
                raise NotImplementedError()

    def _dump_mmd_icdf_coefficients_df(self, df: pd.DataFrame):
        kernel_name = df.loc[0, "kernel"]
        N = df.loc[0, "N"]
        configuration = dict2str(self.get_configurations(df))
        match self.df_format:
            case "parquet":
                df.to_parquet(self.approx_mmd_dir / f"mmd_icdf__configuration={configuration}__kernel={kernel_name}__N={N}.parquet")
            case _:
                raise NotImplementedError()

    def _dump_nstar_df(self):
        if self.df_nstar is None:
            warnings.warn("No df_nstar to dump. Run generalizability_analysis to initialize it.")
            return

        match self.df_format:
            case "parquet":
                self.df_nstar.to_parquet(self.outputs_dir / f"nstar.{self.df_format}")
            case _:
                raise NotImplementedError()

    def _get_existing_precomputed_mmd(self, configuration: dict, kernel_obj: kernels.base.Kernel, N: int):
        if self.dfmmd is not None:
            precomputed_mmd = self.dfmmd.loc[self.dfmmd["kernel"] == str(kernel_obj)]
            precomputed_mmd = precomputed_mmd.query("N == @N")
            for factor, lvl in configuration.items():
                precomputed_mmd = precomputed_mmd.query(f"{factor} == @lvl")
        else:
            precomputed_mmd = pd.DataFrame()

        return precomputed_mmd

    # ---- Routines to compute/estimate/approximate the MMD
    def _estimate_mmd_from_experiments_rankings(self, sample: ru.SampleAM, configuration: dict,
                                                kernel_obj: kernels.rankings.RankingKernel, N: int,
                                                method: Literal["naive", "vectorized", "embedding", "approximation"]):

        # Get a subsample of size N
        distr = du.PMFDistribution.from_sample(sample)
        sample = distr.sample(N)

        precomputed_mmd = self._get_existing_precomputed_mmd(configuration, kernel_obj, N)

        if not precomputed_mmd.empty:
            return precomputed_mmd
        else:
            dfmmd = kernel_obj.mmd_distribution_many_n(sample=sample, nmin=2, nmax=N // 2, step=2,
                                                       rep=self.config_params["rep"],
                                                       disjoint=self.config_sampling["disjoint"],
                                                       replace=self.config_sampling["replace"],
                                                       method=method,
                                                       N=N, use_cached_support_matrix=True)

            for factor, lvl in configuration.items():
                dfmmd.loc[:, factor] = lvl

            if self.dump_results:
                self._dump_sample_mmd_df(dfmmd)

            return dfmmd

    def _estimate_mmd__from_experiments_vectors(self, s: np.ndarray[float], configuration: dict,
                                                kernel_obj: kernels.vectors.VectorKernel, N: int,
                                                method: Literal["naive", "embedding", "approximation"], seed: int = None):

        # Get a subsample of size N
        rng = np.random.default_rng(seed=seed)
        s = rng.choice(s.T, size=N).T

        precomputed_mmd = self._get_existing_precomputed_mmd(configuration, kernel_obj, N)

        if not precomputed_mmd.empty:
            return precomputed_mmd
        else:
            dfmmd = kernel_obj.mmd_distribution_many_n(s=s, nmin=2, nmax=N // 2, step=2,
                                                       rep=self.config_params["rep"],
                                                       disjoint=self.config_sampling["disjoint"],
                                                       replace=self.config_sampling["replace"],
                                                       method=method,
                                                       N=N, use_cached_support_matrix=True)

            for factor, lvl in configuration.items():
                dfmmd.loc[:, factor] = lvl

            if self.dump_results:
                self._dump_sample_mmd_df(dfmmd)

        return dfmmd

    def estimate_mmd(self, sample: Union[ru.SampleAM, np.ndarray[float]], configuration: dict,
                     kernel_obj: kernels.base.Kernel, N: int,
                     method: Literal["naive", "vectorized", "embedding", "approximation"]):

        # Detect if rankings or vectors
        if isinstance(sample, ru.SampleAM) and isinstance(kernel_obj, kernels.rankings.RankingKernel):
            return self._estimate_mmd_from_experiments_rankings(sample, configuration, kernel_obj, N, method)
        elif isinstance(sample, np.ndarray) and isinstance(kernel_obj, kernels.vectors.VectorKernel):
            return self._estimate_mmd__from_experiments_vectors(sample, configuration, kernel_obj, N, method)
        else:
            raise TypeError(f"Parameter sample with type {type(sample)} is not a valid input type.")

    def _estimate_nstar_from_experiments(self, sample: Union[ru.SampleAM, np.ndarray[float]],
                                         configuration: dict,
                                         kernel_obj: kernels.base.Kernel, N: int,
                                         method: Literal["naive", "vectorized", "embedding"] = "embedding"):

        dfmmd = self.estimate_mmd(sample, configuration, kernel_obj, N, method)

        dfq = (dfmmd.groupby("n")["mmd"].quantile(self.config_params["alpha"], interpolation="higher")
               .rename("q_alpha").rename_axis(index=["n", "alpha"]).reset_index())

        out = []
        for alpha, group in dfq.groupby("alpha").groups.items():
            dftmp = dfq.iloc[group]
            logq = np.log(dftmp["q_alpha"].values.reshape(-1, 1))
            logn = np.log(dftmp["n"].values.reshape(-1, 1))

            # logn = b1 * logq + b0
            lr = LinearRegression()
            lr.fit(logq, logn)
            b1 = lr.coef_[0, 0]
            b0 = lr.intercept_[0]

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
    def wilson_hilferty_inv(z: np.ndarray, k:float, a: float) -> np.ndarray[float]:
        return a * k * (np.sqrt(2/(9*k)) * z + 1 - 2/(9*k)) ** 3

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

    def _mmd_cdf_approximation(self, eps: float, L1: float, L4: float, n: int) -> float:
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

        return self.normal_cdf_Lin(self.wilson_hilferty(n*eps**2, k, a))

    def _mmd_icdf_approximation(self, alpha: np.ndarray[float], L1, L4, n) -> np.ndarray[float]:

        a = (L4 - L1 ** 2) / L1
        k = 2 * L1 ** 2 / (L4 - L1 ** 2)

        return np.sqrt(self.wilson_hilferty_inv(self.normal_icdf_Lin(alpha), k, a)  / n)

    @staticmethod
    def _get_nstar_mmd_icdf_approximation(eps: float, alpha: float, L1: float, L4: float) -> float:
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

        return -2 * np.log(eps) + 3 * np.log(np.sqrt(2 / (9 * k)) * (c0 + c1 * np.sqrt(c2 - c3 * np.log(2 * (1 - alpha)))) + (1 - 2 / (9 * k))) + np.log(a * k)

    # --- Approximation of nstar

    def _estimate_nstar_from_approximation(self, sample: Union[ru.SampleAM, np.ndarray[float]], configuration: dict,
                                           kernel_obj: kernels.base.Kernel, N: int = None):
        """
        use the approximation of the MMD using the approximate CDF of a normal variable. Details
            in the paper Matteucci et al. (2025).

        Returns
        -------

        Literature
        ----------
        Matteucci et al. (2025): Matteucci, Federico, et al. "Generalizability of experimental studies." arXiv preprint arXiv:2406.17374 (2024).
        Lin (1989): Lin, Jinn‐Tyan. "Approximating the normal tail probability and its inverse for use on a pocket calculator." Journal of the Royal Statistical Society: Series C (Applied Statistics) 38.1 (1989): 69-70.
        Choudhury (2007) : Choudhury, Amit, Subhasis Ray, and Pradipta Sarkar. "Approximating the cumulative distribution function of the normal distribution." Journal of Statistical Research 41.1 (2007): 59-67.
        """

        alphas = np.array(self.config_params["alpha"])
        deltas = np.array(self.config_params["delta"])

        if np.min(alphas) < 0.6:
            warnings.warn("The approximation of the MMD might not be reliable for alpha < 0.6.")

        if isinstance(sample, ru.SampleAM) and isinstance(kernel_obj, kernels.rankings.RankingKernel):
            if N is not None:
                distr = du.PMFDistribution.from_sample(sample)
                sample = distr.sample(N)

            support, pmf = sample.get_support_pmf()
            x = kernel_obj._convert_sample_to_input_format(support)
            K = kernel_obj.gram_matrix(x, x)
            m = len(support)

        elif isinstance(sample, np.ndarray) and isinstance(kernel_obj, kernels.vectors.VectorKernel):
            if N is not None:
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

        out = []
        for alpha, delta in product(alphas, deltas):
            eps = kernel_obj.get_eps(delta, na=self.na)
            nstar = np.exp(self._get_nstar_mmd_icdf_approximation(eps, alpha, L1, L4))

            result_dict = dict(configuration,
                               **dict(kernel=str(kernel_obj), alpha=alpha, eps=eps, delta=delta,
                                      disjoint=None, replace=None,
                                      method="approximation", N=N, nstar=nstar))
            out.append(result_dict)

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

    def _generalizability_analysis_one_configuration(self, sample_rankings: ru.SampleAM,
                                                     sample_vectors: np.ndarray[float], out: List = None,
                                                     configuration: dict = None):

        configuration = configuration if configuration is not None else dict()
        out = out if out is not None else []

        # Loop over the kernels
        for kernel_obj in self.kernels:
            # Set the Universe
            kernel_obj.set_support(sample_rankings.get_support_pmf()[0])

            for N in range(self.config_sampling["sample_size"], len(sample_rankings),
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

    def generalizability_analysis(self):

        if self.verbose:
            iterator = tqdm(self._get_configurations_and_grouped_df(),
                            position=0, desc="Configurations", leave=True)
        else:
            iterator = self._get_configurations_and_grouped_df()

        out = []
        for configuration, df_grouped in iterator:
            # Rank the alternatives
            results_rankings = ru.get_matrix_from_df(df_grouped, factors=list(self.all_factors),
                                                     alternatives=self.config_data["alternatives_col_name"],
                                                     target=self.config_data["target_col_name"],
                                                     get_rankings=True,
                                                     lower_is_better=self.config_data["target_is_error"],
                                                     impute_missing=True,
                                                     as_numpy=True)
            results_matrix = ru.get_matrix_from_df(df_grouped, factors=list(self.all_factors),
                                                   alternatives=self.config_data["alternatives_col_name"],
                                                   target=self.config_data["target_col_name"],
                                                   get_rankings=False, impute_missing=True, as_numpy=True)

            # Global sample of rankings available
            rankings_all = ru.SampleAM.from_rank_vector_matrix(np.array(results_rankings))
            out = self._generalizability_analysis_one_configuration(sample_rankings=rankings_all,
                                                                    sample_vectors=results_matrix, out=out,
                                                                    configuration=configuration)

        self.df_nstar = pd.DataFrame(out)

        if self.dump_results:
            self._dump_nstar_df()

        return self.df_nstar

class PlotManager(ProjectManager):
    def __init__(self, config_yaml_path: Union[str, Path], save: bool = True, show: bool = True):
        super().__init__(config_yaml_path, is_project_manager=False)

        self.show = show
        self.save = save
        self.boxplot_args = None
        self.pretty_kernels = None
        self.pretty_columns = None
        self.df_nstar = None
        self.dfmmd = None

        self._load_nstar_df()
        self._add_Nmax_column_to_dfnstar()
        self._add_latex_column_to_dfnstar()
        self._load_precomputed_mmd_df()
        self._load_mmd_icdf_coefficients_df()
        self._load_preconfigured_plotting_parameters()

    def _add_Nmax_column_to_dfnstar(self):
        self.df_nstar = self.df_nstar.join(self.df_nstar.groupby(self.configuration_factors)["N"].max(),
                                           on=self.configuration_factors, rsuffix="max")

    def _add_latex_column_to_dfnstar(self):
        self.df_nstar.loc[:, "kernel_latex"] = self.df_nstar["kernel"].apply(
            lambda x: Kernel.from_string(x).latex_str())

    def _load_preconfigured_plotting_parameters(self):
        sns.set(style="ticks", context="paper", font="times new roman")

        # mpl.use("TkAgg")
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = r"""
            \usepackage{mathptmx}
            \usepackage{amsmath}
        """
        mpl.rc('font', family='Times New Roman')

        # pretty names
        self.pretty_columns = {"alpha": r"$\alpha^*$", 'eps': r"$\varepsilon^*$", 'nstar': r"$n^*$",
                               'delta': r"$\delta^*$",
                               'N': r"$N$", 'nstar_absrel_error': "relative error", 'aq': r"$\varepsilon^\alpha_n$",
                               'n': r"$n$"}  # columns

        self.pretty_kernels = {"borda_kernel_idx_OHE": r"$\kappa_b^{\text{OHE}, 1/n}$",
                               "mallows_kernel_nu_auto": r"$\kappa_m^{1/\binom{n}{2}}$",
                               "jaccard_kernel_k_1": r"$\kappa_j^{1}$"}  # kernels
        self.pretty_kernels.update({"borda_kernel_idx_OHE": "$g_1$", "mallows_kernel_nu_auto": "$g_3$",
                                    "jaccard_kernel_k_1": "$g_2$"})  # rename to goal_1, 2, 3

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

    def plot_nstar_on_alpha_delta(self, alpha_fixed: float = 0.95, delta_fixed: float = 0.05, fig_width: float = 6.5,
                                  close_other_plots: bool = True):

        if close_other_plots:
            plt.close("all")
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_width / 2.5), width_ratios=(1, 1), sharey=True)

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
        plt.subplots_adjust(wspace=.12, top=0.86)

        fig.legend(handles=handles, labels=labels, bbox_to_anchor=(0, 0.82 + 0.02, 1, 0.2),
                   loc="upper left", mode="expand", borderaxespad=1, ncol=5, frameon=False)

        ax.set_yscale("log")

        sns.despine(right=True, top=True)
        if self.save:
            plt.savefig(self.figures_dir / "encoders_nstar_alpha_delta.pdf")
        if self.show:
            plt.show()

    def plot_simulated_experimental_study(self, configuration: dict, alpha: float, delta: float, fig_width: float = 6.5,
                                          close_other_plots: bool = True, kernels: List = None):
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

        numN = self.dfmmd.nunique()["N"]

        kernels = kernels if kernels is not None else self.kernels
        for kernel_obj in kernels:

            eps = kernel_obj.get_eps(delta, na=self.na)

            fig, axes = plt.subplots(3, numN, figsize=(fig_width, 0.7 * fig_width), sharex=False, sharey="row",
                                     layout="constrained",
                                     height_ratios=[7, 7, 1])

            padding = 0.8
            xmin = eps * padding
            xmax = max(eps, self.dfmmd["mmd"].max()) / padding

            for icol, Ncol in enumerate(self.dfmmd["N"].unique()):

                query_str = self._get_query_string_from_configuration(configuration)

                dfplot = (self.dfmmd.query(query_str)
                          .query("kernel == @kernel_obj.__str__()").query("N == @Ncol"))

                dfq = (dfplot.groupby("n")["mmd"].quantile(self.config_params["alpha"], interpolation="higher")
                       .rename("q_alpha").rename_axis(index=["n", "alpha"]).reset_index())

                alpha_quantiles = {
                    n: np.quantile(np.quantile(dfq.query("n == @n")["q_alpha"], alpha, method="linear"), alpha)
                    for n in dfq["n"].unique()}
                dfaq = pd.DataFrame(alpha_quantiles, index=[0]).melt(var_name="n", value_name="aq").rename(
                    columns=self.pretty_columns)

                # -- Generalizability (MMD cdf)
                ax = axes[0, icol]

                ax.set_title(f"$N = {Ncol}$")
                ax.set_xlim(xmin, xmax)
                ax.set_xscale("log")

                ax.axhline(alpha, **self.axlines_args)
                ax.axvline(eps, **self.axlines_args)

                ax.set_ylabel(r"$n$-Gen / Pr(MMD$_n < \varepsilon$)")

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    sns.ecdfplot(dfplot, x="mmd", hue="n", ax=ax, legend=False, **self.lineplot_args)

                # Clean after seaborn
                ax.set_xlabel("")
                ax.set_xticklabels([])

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
                lr = LinearRegression()
                lr.fit(X, y)
                ns_pred = np.exp(lr.predict(np.log(epss).reshape(-1, 1)).reshape(1, -1)[0])
                nstar = int(ns_pred[np.argmin(np.abs(epss - eps))])

                ax.plot(epss, ns_pred, color="maroon")
                ax.plot(eps, nstar, marker='*', color='maroon', markersize=7)
                ax.text(eps * 2, 1.5 * nstar, rf"$n^*_{{{Ncol}}}$", color="maroon")

                # Quantile lines
                # for (n, aq), color in zip(alpha_quantiles.items(), sns.color_palette(palette, n_colors=len(alpha_quantiles))):
                #     ax.vlines(aq, ymin=n, ymax=nstar, ls="-", color=color, lw=lw)
                # ax.axvline(aq, ymin=1, ymax=1.2, ls=":", color=color, lw=lw, zorder=-1, clip_on=False)
                ax.set_xlim(xmin, xmax)

                # Turn off unnecessary axes (they're here to be replaced by the colormap)
                ax = axes[2, icol]
                ax.axis("off")

                # Add colormap
                if Ncol == self.dfmmd["N"].max():
                    sm = plt.cm.ScalarMappable(cmap="crest_r",
                                               norm=plt.Normalize(self.dfmmd["n"].min(), self.dfmmd["n"].max()))
                    ax.figure.colorbar(sm, ax=axes[-1, :], location="bottom", shrink=0.5, extend="max", label="$n$",
                                       pad=0,
                                       fraction=1, ticks=range(0, self.dfmmd["N"].max(), 2))

            # - General formatting
            sns.despine(top=True, right=True)

            if self.save:
                plt.savefig(self.figures_dir / f"simulated_study__kernel={kernel_obj}.pdf")
            if self.show:
                plt.show()

    def plot_nstar_method_comparison(self, ):

        plt.close("all")

        for kernel_obj in self.kernels:
            tmp = self.df_nstar.query("kernel == @kernel_obj.__str__()").drop(columns=["disjoint", "replace"])
            tmp_emb = tmp.query("method != 'approximation'").drop(columns="method")
            tmp_emb = tmp_emb.set_index([col for col in tmp_emb.columns if col != "nstar"])
            tmp_app = tmp.query("method == 'approximation'").drop(columns="method")
            tmp_app = tmp_app.set_index([col for col in tmp_app.columns if col != "nstar"])

            dfplot = tmp_emb / tmp_app
            dfplot = dfplot.reset_index()

            fig, axes = plt.subplots(1, 2)
            fig.suptitle(kernel_obj)

            ax = axes[0]
            ax.set_title("Comparison embedding and approximation")
            sns.boxplot(data=dfplot, x="alpha", y="nstar", hue="delta", ax=ax, **self.boxplot_args)
            ax.axhline(1, color="grey", ls="--")
            ax.set_ylabel("emb/app")

            ax = axes[1]
            ax.set_title("Median emb/app. Variation on the fixed configuration")
            dfplot2 = dfplot.groupby(["alpha", "delta"])["nstar"].agg(lambda x: np.median(np.abs(x))).reset_index()
            sns.scatterplot(data=dfplot2, x="alpha", y="delta", hue="nstar", palette="vlag", size="nstar", ax=ax)

            # plt.get_current_fig_manager().window.state('zoomed')
            fig.show()

    def plot_nstar_approximation_comparison(self, configuration: dict, kernel_name: str = None,
                                            close_other_plots: bool = True):

        query_str = self._get_query_string_from_configuration(dict(configuration, **{"N": self.dfmmd["N"].max(),
                                                                                     "kernel": kernel_name}))

        mmd_cdf_symbol = r"$\hat F_n$"
        approx_cdf_symbol = r"$\sim \Phi_n$"

        dfmmd1 = self.dfmmd.query(query_str)
        dfcoef1 = self.icdf_coefficiens.query(query_str)

        if close_other_plots:
            plt.close("all")

        fig, ax = plt.subplots()
        for n in dfmmd1["n"].unique()[::2]:
            dfmmd2 = dfmmd1.query("n == @n")

            xmin = np.quantile(dfmmd2["mmd"], 0.6)
            xmax = dfmmd2["mmd"].max()

            L1, L4 = dfcoef1[["L1", "L4"]].values.flatten()
            # a = (L4 - L1 ** 2) / L1
            # k = 2 * L1 ** 2 / (L4 - L1 ** 2)
            # r = 1

            # Y
            # Y = a * rng.chisquare(df=k, size=1000) ** r

            # close formula
            epss = np.linspace(xmin, xmax, 1000)
            CF = self._mmd_cdf_approximation(epss, L1, L4, n)

            sns.ecdfplot(data=dfmmd2, x="mmd", ax=ax, c="slategray", label=mmd_cdf_symbol)
            # sns.ecdfplot(x=np.sqrt(Y) / np.sqrt(n), ax=ax, label="Y", ls=":", c="orange")
            sns.lineplot(x=epss, y=CF, ax=ax, label=approx_cdf_symbol, ls="--", c="blue")

        # fix legend
        h, l = ax.get_legend_handles_labels()
        h = [h[l.index(s)] for s in [mmd_cdf_symbol, approx_cdf_symbol]]
        l = [mmd_cdf_symbol, approx_cdf_symbol]
        ax.legend(h, l, frameon=False)

        ax.set_xscale("log")
        ax.set_ylabel("\hat\Phi_n")
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_xlim(10e-3, 2)

        sns.despine(top=True, right=True)

        fig.show()

if __name__ == "__main__()":
    pm = ProjectManager(config_yaml_path="config.yaml")
    df_nstar = pm.generalizability_analysis()
