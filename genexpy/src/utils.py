import os
from functools import reduce
from itertools import product
from pathlib import Path
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
import yaml

from genexpy import kernels as kcu


def load_class(path):
    _, cls = path.rsplit(".", maxsplit=1)
    return getattr(kcu, cls)



class ProjectManager:
    """
    Manages the directories, loading the data from config and precomputed results, as well as

    """

    df_format = "parquet"

    def __init__(self):
        self.config_kernels = None
        self.config_sampling = None
        self.config_data = None
        self.config_params = None
        self.load_existing_results_flag = False
        self.figures_dir = None
        self.outputs_dir = None
        self.mmd_dir = None

    def move_to_script_directory(self):
        try:
            os.chdir("demos/Template")
        except FileNotFoundError:
            pass

    def load_config_file(self, config_yaml_path):

        # 1. Load the config file

        with open(config_yaml_path, 'r') as file:
            config = yaml.safe_load(file)

        self.outputs_dir = Path(config["paths"]["outputs_dir"])
        self.figures_dir = self.outputs_dir / "figures"
        self.mmd_dir = self.outputs_dir / "precomputed_MMD"

        config_data = config["data"]
        config_sampling = config["sampling"]
        config_kernels = dict()  # will be filled later in this functions

        config_params = config["parameters"]
        if isinstance(config_params["alpha"], float):
            config_params["alpha"] = [config_params["alpha"]]
        if isinstance(config_params["delta"], float):
            config_params["delta"] = [config_params["delta"]]

        # 2. Load the dataframe of results

        df = pd.read_parquet(config_data["dataset_path"])

        # TODO adapt code to remove this limitation
        assert sum(value is None for value in config_data['experimental_factors_name_lvl'].values()) == 1, \
            "Exactly one factor must be set to null in config.yaml."

        # Query df for the values of the held-constant factors (specified as NOT '_all' or "null" in config.yaml)
        df = df.query(" and ".join(f"{factor} == '{lvl}'" if isinstance(lvl, str) else f"{factor} == {lvl}"
                                   for factor, lvl in config_data['experimental_factors_name_lvl'].items()
                                   if lvl not in [None, "_all"])).reset_index(drop=True)

        # Get the combinations of levels of design factors
        # Each combination has an independent generalizability analysis
        try:
            config_data["design_factor_combinations"] = df.groupby(
                [factor for factor, lvl in config_data['experimental_factors_name_lvl'].items() if
                 lvl == "_all"]).groups
        except ValueError:
            config_data["design_factor_combinations"] = {"None": df.index}

        config_data["results"] = df

        self.config_data = config_data
        self.all_factors = self.config_data["experimental_factors_name_lvl"]
        self.generalizability_factors = {factor: lvl for factor, lvl in self.config_data["experimental_factors_name_lvl"].items() if lvl is None}
        self.held_constant_factors = {factor: lvl for factor, lvl in self.config_data["experimental_factors_name_lvl"].items() if lvl not in ["_all", None]}
        self.design_factors = {factor: lvl for factor, lvl in self.config_data["experimental_factors_name_lvl"].items() if lvl == "_all"}

        self.na = df.nunique()[self.config_data["alternatives_col_name"]]

        # 3. Load the kernels
        for kernel_dict in config["kernels"]:
            kernel_name_base = kernel_dict["kernel"]
            kernel_params = kernel_dict["params"]
            kernel_cls = load_class(kernel_dict["class_impl"])

            param_combinations = product(*[[(param, val) for val in vals] for param, vals in kernel_params.items()])
            for pc in param_combinations:

                pc = dict(pc)

                # find the index of the selcted alternative
                if kernel_name_base == "borda_kernel":
                    pc["idx"] = df[config_data["alternatives_col_name"]].unique().tolist().index(pc["alternative"])
                    del pc["alternative"]

                kernel_name_inst = kernel_name_base + reduce(lambda x, y: x + y,
                                                             [f"__{p[0]}={p[1]}" for p in pc.items()])
                kernel_obj = kernel_cls(**pc)
                kernel_obj._instantiate_parameters(na=df.nunique()[self.config_data["alternatives_col_name"]])

                # epsstar = kernel_obj.get_eps(config_params["delta"])
                config_kernels[kernel_name_inst] = (kernel_name_base, kernel_obj)

        self.config_params = config_params
        self.config_sampling = config_sampling
        self.kernels = config_kernels

        self.results = self.config_data["results"]


    def load_existing_results(self, kernel_name: str, N: int):
        try:
            match self.df_format:
                case "parquet":
                    return pd.read_parquet(self.mmd_dir / f"mmd__kernel={kernel_name}__N={N}.{self.df_format}")
                case _:
                    raise NotImplementedError()
        except FileNotFoundError:
            return None

    def create_project_directories(self):
        self.mmd_dir.mkdir(parents=True, exist_ok=True)
        readme = self.mmd_dir / "README.md"
        readme.write_text("""
            This directory contains the precomputed distributions of the MMD. 
        """)

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        readme = self.figures_dir / "README.md"
        readme.write_text("""
            This directory contains the figures and plots. 
        """)



    def get_factor_levels(self, idf):
        a = dict(self.design_factors, **self.held_constant_factors)

        # Check that design and held-constant factors have been filtered correctly. They should have exactly one unique values
        if (idf.nunique()[a.keys()] > 1).any():
            raise ValueError("Factor levels not unique after query.")

        # Current levels of design and held-constant factor
        self.factors_dict = dict(a, **{factor: idf[factor].unique()[0] for factor, _ in a.items()})

    def dump_mmd_dataset(self, df: pd.DataFrame):
        # num_existing_dataframes = len(list(self.mmd_dir.glob(f"*.{self.df_format}")))
        kernel_name = df.loc[0, "kernel"]
        N = df.loc[0, "N"]
        match self.df_format:
            case "parquet":
                df.to_parquet(self.mmd_dir / f"mmd__kernel={kernel_name}__N={N}.{self.df_format}")
            case _:
                raise NotImplementedError()

    def dump_output_dataset(self, df: pd.DataFrame):
        match self.df_format:
            case "parquet":
                df.to_parquet(self.mmd_dir / f"nstar.{self.df_format}")
            case _:
                raise NotImplementedError()


    def estimate_mmd(self, rankings_all, kernel_obj, N):

        sample = rankings_all.get_subsample(N, seed=N)

        if self.load_existing_results_flag:
            dfmmd = self.load_existing_results(kernel_name=str(kernel_obj), N=N)
        else:
            dfmmd = None

        if dfmmd is None:
            dfmmd = kernel_obj.mmd_distribution_many_n(sample=sample, nmin=2, nmax=N // 2, step=2,
                                                       rep=self.config_params["rep"],
                                                       disjoint=self.config_sampling["disjoint"],
                                                       replace=self.config_sampling["replace"],
                                                       N=N, use_cached_universe_matrix=True)

            for factor, lvl in self.factors_dict.items():
                dfmmd.loc[:, factor] = lvl

            self.dump_mmd_dataset(dfmmd)

        self.dfmmd = dfmmd

        return dfmmd


    def estimate_nstar(self, kernel_obj, N, out):

        dfq = (self.dfmmd.groupby("n")["mmd"].quantile(self.config_params["alpha"], interpolation="higher")
               .rename("q_alpha").rename_axis(index=["n", "alpha"]).reset_index())

        for alpha, group in dfq.groupby("alpha").groups.items():
            dftmp = dfq.iloc[group]
            logq = np.log(dftmp["q_alpha"].values.reshape(-1, 1))
            logn = np.log(dftmp["n"].values.reshape(-1, 1))

            # logn = b1 * logq + b0
            lr = LinearRegression().fit(logq, logn)
            b1 = lr.coef_[0, 0]
            b0 = lr.intercept_[0]

            for delta in self.config_params["delta"]:
                eps = kernel_obj.get_eps(delta, na=self.na)

                nstar = np.exp(b1 * np.log(eps) + b0)

                result_dict = dict(self.factors_dict,
                                   **dict(kernel=str(kernel_obj), alpha=alpha, eps=eps, delta=delta,
                                          disjoint=self.config_sampling["disjoint"],
                                          replace=self.config_sampling["replace"], N=N, nstar=nstar))
                out.append(result_dict)
        return out











