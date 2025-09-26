"""
Run the generalizability analysis as specified in config.yaml.

test file for the strong refactoring: hide as much as possible under the hood

TODO give possibility to copy the template directory directly
    maybe develop a helper script to fill in the vcarious fields of config

TODO option to run from console

TODO option to run from notebook?

1. Specifying the analysis we want
    - config.yaml

2. Loading from config

3. For every combination of design factors, compute generalizability


"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from tqdm.auto import tqdm

from genexpy import utils as u
from genexpy import rankings_utils as ru
from genexpy import kernels as ku

pm = u.ProjectManager()
pm.move_to_script_directory()
pm.load_config_file("config.yaml")
pm.create_project_directories()


rv1 = np.array([[0, 2, 2, 3, 0],
                [2, 1, 1, 0, 2],
                [2, 1, 1, 2, 2],
                [1, 0, 0, 1, 1]])

rv2 = np.array([[3, 3, 0, 2, 3],
                [0, 0, 2, 1, 0],
                [2, 2, 2, 1, 2],
                [1, 1, 1, 0, 1]])

# s1 = ru.SampleAM.from_rank_vector_matrix(rv1)
# s2 = ru.SampleAM.from_rank_vector_matrix(rv2)

k = ku.JaccardKernel(k=1)

rv = np.array([[0, 2, 2, 1],
               [2, 1, 1, 0],
               [3, 0, 2, 1]]).T


#%%%




out = []
for fixed_levels, idf in tqdm(list(pm.results.groupby(list(dict(pm.design_factors, **pm.held_constant_factors).keys()))), position=0, desc="Configurations", leave=True):

    pm.get_factor_levels(idf)

    # Rank the alternatives
    rank_matrix = ru.get_rankings_from_df(idf, factors=list(pm.config_data["experimental_factors_name_lvl"].keys()),
                                          alternatives=pm.config_data["alternatives_col_name"],
                                          target=pm.config_data["target_col_name"],
                                          lower_is_better=False, impute_missing=True)

    # Global sample of rankings available
    rankings_all = ru.SampleAM.from_rank_vector_matrix(rank_matrix.to_numpy())

    # Loop over the kernels
    for kernel_name_inst, (kernel_name_base, kernel_obj) in pm.kernels.items():

        # Set the Universe
        kernel_obj.set_universe(rankings_all.get_universe_pmf()[0])
        kernel_obj._instantiate_parameters(na=rankings_all.get_na())

        for N in range(pm.config_sampling["sample_size"], len(rankings_all), pm.config_sampling["sample_size"]):
            dfmmd = pm.estimate_mmd(rankings_all, kernel_obj, N)
            out = pm.estimate_nstar(kernel_obj, N, out)

# Store nstar predictions
out = pd.DataFrame(out)
out.to_parquet(pm.outputs_dir / "nstar.parquet")

# %% Plots

import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt

reload(u)

class PlotManager(u.ProjectManager):
    def __init__(self):
        super().__init__()

        sns.set(style="ticks", context="paper", font="times new roman")

        # mpl.use("TkAgg")
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = r"""
            \usepackage{mathptmx}
            \usepackage{amsmath}
        """
        mpl.rc('font', family='Times New Roman')

        # pretty names
        self.pc = {"alpha": r"$\alpha^*$", "eps": r"$\varepsilon^*$", "nstar": r"$n^*$", "delta": r"$\delta^*$", "N": r"$N$",
              "nstar_absrel_error": "relative error", "aq": r"$\varepsilon^\alpha_n$", "n": r"$n$"}  # columns
        self.pk = {"borda_kernel_idx_OHE": r"$\kappa_b^{\text{OHE}, 1/n}$",
              "mallows_kernel_nu_auto": r"$\kappa_m^{1/\binom{n}{2}}$",
              "jaccard_kernel_k_1": r"$\kappa_j^{1}$"}  # kernels
        self.pk.update({"borda_kernel_idx_OHE": "$g_1$", "mallows_kernel_nu_auto": "$g_3$",
                   "jaccard_kernel_k_1": "$g_2$"})  # rename to goal_1, 2, 3

        self.boxplot_args = dict(
            showfliers=False, palette="cubehelix",
            dodge=True, native_scale=False, fill=False, width=0.75, boxprops={"linewidth": 1.2}, gap=0.25
        )



    def load_nstar(self):
        match self.df_format:
            case "parquet":
                self.df_nstar = pd.read_parquet(self.outputs_dir / f"nstar.{self.df_format}")
            case _:
                raise NotImplementedError()

        # TODO: move a somewhere else
        a = dict(self.design_factors, **self.held_constant_factors)
        self.df_nstar = self.df_nstar.join(self.df_nstar.groupby(list(a.keys()))["N"].max(), on=list(a.keys()), rsuffix="max")


    def load_intermediate_mmd(self):
        self.mmd_dir = self.outputs_dir / "precomputed_MMD"
        try:
            match self.df_format:
                case "parquet":
                    dfs = [pd.read_parquet(filepath) for filepath in self.mmd_dir.glob("*.parquet")]
                case _:
                    raise NotImplementedError()
        except FileNotFoundError:
            return None

        self.dfmmd = pd.concat(dfs, ignore_index=True)

    def plot_nstar_on_alpha_delta(self, alpha_fixed: float = 0.95, delta_fixed: float = 0.05):

        plt.close("all")
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 5.5 / 2.5), width_ratios=(1, 1), sharey=True)

        # ----  ALPHA
        ax = axes[0]
        dfplot = self.df_nstar.loc[(self.df_nstar["delta"] == delta_fixed) & (self.df_nstar["N"] == self.df_nstar["Nmax"])]

        # Make dfplot pretty
        dfplot = dfplot.rename(columns=self.pc)
        # dfplot["kernel"] = dfplot["kernel"].map(self.pk)

        sns.boxplot(dfplot, x=self.pc["alpha"], y=self.pc["nstar"], ax=ax, hue="kernel", legend=False, **self.boxplot_args)
        ax.grid(color="grey", alpha=0.2)

        # ----  DELTA
        ax = axes[1]
        dfplot = self.df_nstar.loc[(self.df_nstar["alpha"] == alpha_fixed) & (self.df_nstar["N"] == self.df_nstar["Nmax"])]

        # Make dfplot pretty
        dfplot = dfplot.rename(columns=self.pc)
        # dfplot["kernel"] = dfplot["kernel"].map(self.pk)

        sns.boxplot(dfplot, x=self.pc["delta"], y=self.pc["nstar"], ax=ax, hue="kernel", legend=True, **self.boxplot_args)
        ax.grid(color="grey", alpha=0.2)

        ax.legend(*ax.get_legend_handles_labels()).get_frame().set_edgecolor("w")

        ax.set_yscale("log")

        sns.despine(right=True, top=True)
        plt.tight_layout(pad=.5)
        plt.subplots_adjust(wspace=.12)
        plt.savefig(self.figures_dir / "encoders_nstar_alpha_delta.pdf")
        plt.show()

    def plot_nstar_prediction(self, alpha, delta):

        plt.close("all")

        palette = "crest_r"
        lw = 0.5

        for kernel_name_inst, (kernel_name_base, kernel_obj) in self.kernels.items():

            eps = kernel_obj.get_eps(delta, na=self.na)
            dfplot = dfmmd.query("kernel == @kernel_obj")

            dfq = (dfplot.groupby("n")["mmd"].quantile(self.config_params["alpha"], interpolation="higher")
                   .rename("q_alpha").rename_axis(index=["n", "alpha"]).reset_index())

            alpha_quantiles = {
                n: np.quantile(np.quantile(dfq.query("n == @n")["q_alpha"], alpha, method="linear"), alpha)
                for n in
                dfq["n"].unique()}
            dfaq = pd.DataFrame(alpha_quantiles, index=[0]).melt(var_name="n", value_name="aq")#.rename(columns=pc)

            fig, axes = plt.subplots(2, 1, figsize=(2.5, 4), sharex=True)

            # -- Generalizability
            ax = axes[0]
            ax.set_xscale("log")
            sns.ecdfplot(dfplot, x="mmd", hue="n", palette=palette, ax=ax,
                         legend=False)

            # Quantile lines
            # for (n, laq), color in zip(alpha_quantiles.items(), sns.color_palette(palette, n_colors=len(alpha_quantiles))):
            #     ax.vlines(laq, ymin=0, ymax=alpha, ls="-", color=color, lw=lw)
            # ax.axvline(laq, ymin=-1.2, ymax=0, ls=":", color=color, lw=lw, zorder=-1, clip_on=False)
            ax.axhline(alpha, color="black", ls="--", lw=lw)
            ax.axvline(eps, color="black", ls="--", lw=lw)

            # -- Quantiles for regression
            ax = axes[1]
            ax.set_xscale("log")
            ax.set_yscale("log")
            sns.lineplot(dfaq, x=self.pc["aq"], y=self.pc["n"], ax=ax, ls="", marker="o", hue=self.pc["n"], legend=False, palette=palette)

            # Linear regression
            X = np.log(dfaq[self.pc["aq"]]).to_numpy().reshape(-1, 1)
            y = np.log(dfaq[self.pc["n"]]).to_numpy().reshape(-1, 1)

            padding = 1.1
            epss = np.linspace(eps / padding, np.max(dfq["q_alpha"]) * padding, 1000)
            lr = LinearRegression().fit(X, y)
            ns_pred = np.exp(lr.predict(np.log(epss).reshape(-1, 1)).reshape(1, -1)[0])
            nstar = int(ns_pred[np.argmin(np.abs(epss - eps))])
            ax.plot(epss, ns_pred, ls="--", color="grey", lw=lw)

            # Quantile lines
            # for (n, aq), color in zip(alpha_quantiles.items(), sns.color_palette(palette, n_colors=len(alpha_quantiles))):
            #     ax.vlines(aq, ymin=n, ymax=nstar, ls="-", color=color, lw=lw)
            # ax.axvline(aq, ymin=1, ymax=1.2, ls=":", color=color, lw=lw, zorder=-1, clip_on=False)
            ax.axvline(eps, color="black", ls="--", lw=lw)

            # - General formatting
            sns.despine(top=True, right=True)
            plt.tight_layout(pad=0.5)

            plt.savefig(self.figures_dir / f"nstar_prediction__kernel={kernel_obj}__N={N}__.pdf")
            plt.show()

plt.close("all")

plotm = PlotManager()
plotm.load_config_file("config.yaml")
plotm.load_intermediate_mmd()
plotm.load_nstar()
plotm.plot_nstar_on_alpha_delta()
plotm.plot_nstar_prediction(alpha=0.95, delta=0.01)
