"""Generate plots for: 
    - Marginal resilience curves
    - Total investment to meet specific targets 
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.ticker import (MaxNLocator,LinearLocator, MultipleLocator)
import matplotlib.pyplot as plt
from matplotlib import cm
from plot_utils import *
from tqdm import tqdm
tqdm.pandas()

mpl.style.use('ggplot')
mpl.rcParams['font.size'] = 10.
mpl.rcParams['font.family'] = 'tahoma'
mpl.rcParams['axes.labelsize'] = 12.
mpl.rcParams['xtick.labelsize'] = 10.
mpl.rcParams['ytick.labelsize'] = 10.

def main(config):
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    figures_data_path = config['paths']['figures']

    countries = ["dma","grd","lca","vct"]
    countries = ["grd"]
    development_scenarios = ["bau","sdg"]
    development_scenarios_colors = [("#9ebcda","#8856a7"),("#99d8c9","#2ca25f")]
    rcps = ["ssp126","ssp245","ssp585"]
    rcp_colors = ["#8c96c6","#8c6bb1","#88419d"]
    hazards = ["coastal","fluvial_defended","pluvial","cyclone_windspeed"]
    subsectors = ["airports","ports","roads","energy","education","health","wtp","wwtp"]
    subsector_colors = ["#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6"]
    adapt_col = "adaptation_investment"
    new_build_col = "new_build_resilience_investment"
    epochs = [2030,2050]
    width = 0.5
    multiply_factor = 1.0e-6
    for country in countries:
        damage_data = pd.read_excel(
                        os.path.join(output_data_path,
                            "adaptation_outcomes",
                            f"{country}_aggregated_results_for_macroeconomic_analysis",
                            f"{country}_all_assets_risks_investments.xlsx"),
                        sheet_name="bau")
        damage_data = damage_data[
                            (damage_data.epoch == 2023) & (
                            damage_data.hazard != "fluvial_undefended") & (
                            damage_data.rp.isin([1,100]))
                            ]
        hazards = list(set(damage_data.hazard.values.tolist()))
        for subsector in subsectors:
            for hz in hazards:
                df = damage_data[(damage_data.subsector == subsector) & (damage_data.hazard == hz)] 
                ymax = multiply_factor*max(df["adaptation_investment_max"].max(),df["avoided_damage_max"].max())
                if len(df.index) > 0:
                    fig, ax = plt.subplots(1,1,figsize=(8,8),dpi=500)
                    xvals = df["service_resilience_achieved_percentage"]
                    ax.plot(xvals,multiply_factor*df["adaptation_investment_mean"],
                                        marker='o',
                                        color='r',
                                        label="Investment mean")
                    ax.fill_between(xvals,
                                    multiply_factor*df["adaptation_investment_min"],
                                    multiply_factor*df["adaptation_investment_max"],
                                    alpha=0.3,facecolor='r',
                                    label=f"Investment min-max")
                    ax1 = ax.twinx()
                    ax1.plot(xvals,multiply_factor*df["avoided_damage_mean"],
                                    marker='s',
                                    color='g',
                                    label="Avoided damages mean")
                    ax1.fill_between(xvals,
                                    multiply_factor*df["avoided_damage_min"],
                                    multiply_factor*df["avoided_damage_max"],
                                    alpha=0.3,facecolor='g',
                                    label="Avoided damages min-max")
                    ax.set_ylim([0,1.1*ymax])
                    # ax.legend(prop={'size':12,'weight':'bold'})
                    ax.set_ylabel('Adaptation Investments (US$ million)',fontweight='bold',fontsize=15)
                    ax.set_xlabel('Service resilience (%)',fontweight='bold',fontsize=15)
                    ax.set_title(f"{subsector.upper()}-{str(hz.split('_')[0]).capitalize()} marginal resilience curve",
                                fontweight='bold',fontsize=18)
                    ax1.set_ylim([0,1.1*ymax])
                    ax1.set_ylabel('Avoided damages (US$ million)',fontweight='bold',fontsize=15)

                    # ask matplotlib for the plotted objects and their labels
                    lines, labels = ax.get_legend_handles_labels()
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    ax1.legend(lines + lines1, labels + labels1,prop={'size':12,'weight':'bold'},loc='upper left')
                    plt.tight_layout()
                    save_fig(os.path.join(figures_data_path,
                                f"{country}_{subsector}_{hz}_resilience_curve.png"))
                    plt.close()
        for epoch in epochs:
            fig, ax = plt.subplots(1,1,figsize=(8,8),dpi=500)
            offset = 0.0
            ymax = 0
            for idx,(dsc,dsc_color) in enumerate(list(zip(development_scenarios,development_scenarios_colors))):
                if epoch == 2030:
                    damage_data = pd.read_excel(
                                    os.path.join(output_data_path,
                                        "adaptation_outcomes",
                                        "aggregated_results_with_targets",
                                        f"{country}_total_risks_investments_with_resilience_goals.xlsx"),
                                    sheet_name=dsc)
                else:
                    damage_data = pd.read_excel(
                                    os.path.join(output_data_path,
                                        "adaptation_outcomes",
                                        "aggregated_results_with_targets",
                                        f"{country}_total_risks_investments_with_90_percent_resilience.xlsx"),
                                    sheet_name=dsc)
                ymax = max(ymax,max(multiply_factor*(damage_data[f"{adapt_col}_max"] + damage_data[f"{new_build_col}_max"])))
                if dsc == "bau":
                    df = damage_data[(damage_data.rcp.isin(["baseline"])) & (damage_data.epoch.isin([2023])) & (damage_data.rp == 1)]
                    ax.bar([1],multiply_factor*df[f"{adapt_col}_mean"],
                            width=width,
                            color="#a63603",
                            yerr=(multiply_factor*(df[f"{adapt_col}_mean"] - df[f"{adapt_col}_min"]),
                                    multiply_factor*(df[f"{adapt_col}_max"] - df[f"{adapt_col}_mean"])),
                            capsize=5,
                            label="Landslide adaptation investments")
                df = damage_data[(damage_data.rcp.isin(["ssp245"])) & (damage_data.epoch.isin([2030])) & (damage_data.rp > 1)]
                x_vals = np.arange(3.5, 4*len(df.index),4) + offset
                bottom  = np.zeros(len(df.index))
                ax.bar(x_vals,multiply_factor*df[f"{adapt_col}_mean"],
                        width=width,
                        color=dsc_color[0],
                        bottom=bottom,
                        label=f"Other hazards - Existing asset adaptation ({dsc.upper()})")
                bottom += multiply_factor*df[f"{adapt_col}_mean"]
                ax.bar(x_vals,multiply_factor*df[f"{new_build_col}_mean"],
                        width=width,
                        color=dsc_color[1],
                        bottom=bottom,
                        yerr=(multiply_factor*(
                                    df[
                                        f"{adapt_col}_mean"
                                        ] + df[
                                            f"{new_build_col}_mean"
                                            ] - df[
                                                f"{adapt_col}_min"
                                            ] - df[
                                            f"{new_build_col}_min"
                                            ]),
                                multiply_factor*(
                                    df[
                                        f"{adapt_col}_max"
                                        ] + df[
                                            f"{new_build_col}_max"
                                            ] - df[
                                                f"{adapt_col}_mean"
                                            ] - df[
                                            f"{new_build_col}_mean"
                                            ])),
                        capsize=5,
                        label=f"All hazards - New asset adaptation ({dsc.upper()})")
                offset += 0.5

            ax.set_ylim([0,1.3*ymax])
            ax.legend(prop={'size':12,'weight':'bold'},loc="upper right")
            ax.text(
                        0.1,
                        0.95,
                        epoch,
                        horizontalalignment='left',
                        transform=ax.transAxes,
                        size=18,
                        weight='bold')
            # ax.legend(prop={'size':12,'weight':'bold'},bbox_to_anchor=(0.9, -0.2))
            ax.set_ylabel('Adaptation Investments (US$ million)',fontweight='bold',fontsize=15)
            xticks = [1] + list(np.arange(3,16,4, dtype=int) + 0.75)
            ax.set_xticks(xticks,["Landslide","RP 5","RP 10","RP 50","RP 100"],
                            fontsize=15, rotation=45,
                            fontweight="bold")
            plt.tight_layout()
            save_fig(os.path.join(figures_data_path,
                        f"{country}_investments_fixed_goals_{epoch}.png"))
            plt.close()

            fig, ax = plt.subplots(1,1,figsize=(8,8),dpi=500)
            offset = 0.0
            for idx,(dsc,dsc_color) in enumerate(list(zip(development_scenarios,development_scenarios_colors))):
                if epoch == 2030:
                    damage_data = pd.read_excel(
                                    os.path.join(output_data_path,
                                        "adaptation_outcomes",
                                        "aggregated_results_with_targets",
                                        f"{country}_total_risks_investments_with_resilience_goals.xlsx"),
                                    sheet_name=dsc)
                else:
                    damage_data = pd.read_excel(
                                    os.path.join(output_data_path,
                                        "adaptation_outcomes",
                                        "aggregated_results_with_targets",
                                        f"{country}_total_risks_investments_with_90_percent_resilience.xlsx"),
                                    sheet_name=dsc)
                ymax = multiply_factor*damage_data["avoided_damage_max"].max()
                if dsc == "bau":
                    df = damage_data[(damage_data.rcp.isin(["baseline"])) & (damage_data.epoch.isin([2023])) & (damage_data.rp == 1)]
                    ax.bar([1],multiply_factor*df["avoided_damage_mean"],
                            width=width,
                            color="#a63603",
                            yerr=(multiply_factor*(df[f"avoided_damage_mean"] - df[f"avoided_damage_min"]),
                                    multiply_factor*(df[f"avoided_damage_max"] - df[f"avoided_damage_mean"])),
                            capsize=5,
                            label="Landslide - Avoided damages")
                df = damage_data[(damage_data.rcp.isin(["ssp245"])) & (damage_data.epoch.isin([2030])) & (damage_data.rp > 1)]
                x_vals = np.arange(3.5, 4*len(df.index),4) + offset
                ax.bar(x_vals,multiply_factor*df[f"avoided_damage_mean"],
                        width=width,
                        color=dsc_color[0],
                        yerr=(multiply_factor*(
                                    df[
                                        f"avoided_damage_mean"
                                        ] - df[
                                            f"avoided_damage_min"
                                            ]),
                                multiply_factor*(
                                    df[
                                        f"avoided_damage_max"
                                        ] - df[
                                            f"avoided_damage_mean"
                                            ])),
                        capsize=5,
                        label=f"Other hazards - Avoided damages ({dsc.upper()})")
                offset += 0.5

            ax.set_ylim([0,1.3*ymax])
            ax.legend(prop={'size':12,'weight':'bold'},loc="upper right")
            ax.text(
                        0.1,
                        0.95,
                        epoch,
                        horizontalalignment='left',
                        transform=ax.transAxes,
                        size=18,
                        weight='bold')
            # ax.legend(prop={'size':12,'weight':'bold'},bbox_to_anchor=(0.9, -0.2))
            ax.set_ylabel('Avoided damages (US$ million)',fontweight='bold',fontsize=15)
            xticks = [1] + list(np.arange(3,16,4, dtype=int) + 0.75)
            ax.set_xticks(xticks,["Landslide","RP 5","RP 10","RP 50","RP 100"],
                            fontsize=15, rotation=45,
                            fontweight="bold")
            plt.tight_layout()
            save_fig(os.path.join(figures_data_path,
                        f"{country}_avoided_damages_fixed_goals_{epoch}.png"))
            plt.close()
            
            



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
