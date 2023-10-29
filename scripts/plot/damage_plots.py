"""Generate plots for: 
    - Aggregated direct damages per climate scenario and RP
    - Aggregated direct damages per climate scenario, hazard and RP
    - Mean direct damages per hazard and sector for landslides (RP=1) and other hazards (RP=100)
    - Matrix of pre-adaptation & post-disaster service resilience for: 
        - Sector 
        - Hazard [landslides (RP=1) and other hazards (RP=100)]
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
    rcps = ["ssp126","ssp245","ssp585"]
    rcp_colors = ["#8c96c6","#8c6bb1","#88419d"]
    hazards = ["coastal","fluvial_defended","pluvial","cyclone_windspeed"]
    subsectors = ["airports","ports","roads","energy","education","health","wtp","wwtp"]
    subsector_colors = ["#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6"]
    epochs = [2030,2050]
    width = 0.5
    multiply_factor = 1.0e-6
    for country in countries:
        damage_data = pd.read_excel(
                        os.path.join(output_data_path,
                            "adaptation_outcomes",
                            "aggregated_results",
                            f"{country}_aggregated_risks_investments.xlsx"),
                        sheet_name="bau")
        damage_data = damage_data[damage_data.existing_asset_adaptation_implemented == "no"]
        rps = sorted(list(set(damage_data.rp.values.tolist())))
        # epochs = sorted(list(set(damage_data.epoch.values.tolist())))

        for epoch in epochs:
            fig, ax = plt.subplots(1,1,figsize=(8,8),dpi=500)
            df = damage_data[(damage_data.rcp.isin(["baseline"])) & (damage_data.epoch.isin([2023])) & (damage_data.rp == 1)]
            ax.bar([1],multiply_factor*df["damage_mean"],
                    width=width,
                    color="#a63603",
                    yerr=(multiply_factor*(df["damage_mean"] - df["damage_min"]),
                            multiply_factor*(df["damage_max"] - df["damage_mean"])),
                    capsize=5,
                    label="Landslide")
            df = damage_data[(damage_data.rcp.isin(["baseline"])) & (damage_data.epoch.isin([2023])) & (damage_data.rp > 1)]
            x_vals = np.arange(3, 4*len(df.index),4, dtype=int)
            ax.bar(x_vals,multiply_factor*df["damage_mean"],
                    width=width,
                    color="#d0d1e6",
                    yerr=(multiply_factor*(df["damage_mean"] - df["damage_min"]),
                            multiply_factor*(df["damage_max"] - df["damage_mean"])),
                    capsize=5,
                    label="Other hazards - Baseline")
            offset = 0.5
            for idx, (rcp,rcp_color) in enumerate(list(zip(rcps,rcp_colors))):
                df = damage_data[(damage_data.rcp.isin([rcp])) & (damage_data.epoch.isin([epoch]))]
                x_vals = np.arange(3, 4*len(df.index),4, dtype=int) + offset
                ax.bar(x_vals,multiply_factor*df["damage_mean"],
                    width=width,
                    color=rcp_color,
                    yerr=(multiply_factor*(df["damage_mean"] - df["damage_min"]),
                            multiply_factor*(df["damage_max"] - df["damage_mean"])),
                    capsize=5,
                    label=f"Other hazards - {rcp.upper()}")
                offset += 0.5

            # ax.legend(prop={'size':12,'weight':'bold'})
            ax.text(
                        0.5,
                        0.95,
                        epoch,
                        horizontalalignment='left',
                        transform=ax.transAxes,
                        size=18,
                        weight='bold')
            ax.legend(prop={'size':12,'weight':'bold'})
            ax.set_ylabel('Direct damages (US$ million)',fontweight='bold',fontsize=15)
            xticks = [1] + list(np.arange(3,16,4, dtype=int) + 0.75)
            ax.set_xticks(xticks,["Landslide","RP 5","RP 10","RP 50","RP 100"],
                            fontsize=15, rotation=45,
                            fontweight="bold")
            plt.tight_layout()
            save_fig(os.path.join(figures_data_path,
                        f"{country}_total_damages_{epoch}.png"))
            plt.close()

        damage_data = pd.read_excel(
                        os.path.join(output_data_path,
                            "adaptation_outcomes",
                            f"{country}_aggregated_results_for_macroeconomic_analysis",
                            f"{country}_total_risks_investments.xlsx"),
                        sheet_name="bau")
        damage_data = damage_data[
                            (damage_data.existing_asset_adaptation_implemented == "no") & (
                            damage_data.hazard != "fluvial_undefended") & (
                            damage_data.rp.isin([1,100]))
                            ]
        ymax = multiply_factor*max(damage_data.damage_max)
        for epoch in epochs:
            fig, ax = plt.subplots(1,1,figsize=(8,8),dpi=500)
            df = damage_data[(damage_data.rcp.isin(["baseline"])) & (damage_data.epoch.isin([2023])) & (damage_data.rp == 1)]
            ax.bar([1],multiply_factor*df["damage_mean"],
                    width=width,
                    color="#a63603",
                    yerr=(multiply_factor*(df["damage_mean"] - df["damage_min"]),
                            multiply_factor*(df["damage_max"] - df["damage_mean"])),
                    capsize=5,
                    label="Landslide")
            df = damage_data[(damage_data.rcp.isin(["baseline"])) & (damage_data.epoch.isin([2023])) & (damage_data.rp > 1)]
            df = df.set_index('hazard')
            df = df.reindex(hazards)
            x_vals = np.arange(3, 4*len(df.index),4, dtype=int)
            ax.bar(x_vals,multiply_factor*df["damage_mean"],
                    width=width,
                    color="#d0d1e6",
                    yerr=(multiply_factor*(df["damage_mean"] - df["damage_min"]),
                            multiply_factor*(df["damage_max"] - df["damage_mean"])),
                    capsize=5,
                    label="RP 100 - Baseline")
            offset = 0.5
            for idx, (rcp,rcp_color) in enumerate(list(zip(rcps,rcp_colors))):
                df = damage_data[(damage_data.rcp.isin([rcp])) & (damage_data.epoch.isin([epoch]))]
                df = df.set_index('hazard')
                df = df.reindex(hazards)
                x_vals = np.arange(3, 4*len(df.index),4, dtype=int) + offset
                ax.bar(x_vals,multiply_factor*df["damage_mean"],
                    width=width,
                    color=rcp_color,
                    yerr=(multiply_factor*(df["damage_mean"] - df["damage_min"]),
                            multiply_factor*(df["damage_max"] - df["damage_mean"])),
                    capsize=5,
                    label=f"RP 100 - {rcp.upper()}")
                offset += 0.5

            # ax.legend(prop={'size':12,'weight':'bold'})
            ax.text(
                        0.5,
                        0.95,
                        epoch,
                        horizontalalignment='left',
                        transform=ax.transAxes,
                        size=18,
                        weight='bold')
            ax.legend(prop={'size':12,'weight':'bold'},loc='upper left')
            ax.set_ylim([0,1.2*ymax])
            ax.set_ylabel('Direct damages (US$ million)',fontweight='bold',fontsize=15)
            xticks = [1] + list(np.arange(3,16,4, dtype=int) + 0.75)
            ax.set_xticks(xticks,["Landslide","Coastal","Fluvial","Pluvial","Cyclone"],
                            fontsize=15, rotation=45,
                            fontweight="bold")
            plt.tight_layout()
            save_fig(os.path.join(figures_data_path,
                        f"{country}_hazard_total_damages_{epoch}.png"))
            plt.close()

        damage_data = pd.read_excel(
                        os.path.join(output_data_path,
                            "adaptation_outcomes",
                            f"{country}_aggregated_results_for_macroeconomic_analysis",
                            f"{country}_all_assets_risks_investments.xlsx"),
                        sheet_name="bau")
        damage_data = damage_data[
                            (damage_data.existing_asset_adaptation_implemented == "no") & (
                            damage_data.epoch == 2023) & (
                            damage_data.hazard != "fluvial_undefended") & (
                            damage_data.rp.isin([1,100]))
                            ]
        damage_totals = damage_data.groupby(["hazard"])["damage_mean"].sum().reset_index()
        ymax = multiply_factor*max(damage_totals.damage_mean)
        fig, ax = plt.subplots(1,1,figsize=(8,8),dpi=500)
        all_hazards = ["landslide"] + hazards
        bottom  = np.zeros(len(all_hazards))
        for idx,(subsector,subsector_color) in enumerate(list(zip(subsectors,subsector_colors))): 
            df = damage_data[damage_data.subsector == subsector]
            if len(df.index) > 0:
                vals = []
                for hz in all_hazards:
                    df1 = df[df.hazard == hz]
                    if len(df1.index) > 0:
                        vals.append(df1.damage_mean.values[0])
                    else:
                        vals.append(0)

                vals = np.array(vals)
                ax.bar(all_hazards,multiply_factor*vals,
                width=width,
                color=subsector_color, 
                label=f"{subsector.upper()}",
                bottom =bottom)
                bottom += multiply_factor*vals

            for c in ax.containers:
                # Optional: if the segment is small or 0, customize the labels
                labels = [round (v.get_height(),1) if v.get_height() > 5.0 else '' for v in c]
                
                # remove the labels parameter if it's not needed for customized labels
                ax.bar_label(c, labels=labels, label_type='center')

        # ax.legend(prop={'size':12,'weight':'bold'},bbox_to_anchor=(1.5, 0.95))
        ax.legend(prop={'size':12,'weight':'bold'})
        ax.set_ylim([0,1.2*ymax])
        ax.set_ylabel('Direct damages (US$ million)',fontweight='bold',fontsize=15)
        # xticks = list(np.arange(1,len(all_hazards),1, dtype=int))
        ax.set_xticklabels(["Landslide","Coastal","Fluvial","Pluvial","Cyclone"],
                        fontsize=15, rotation=45,
                        fontweight="bold")
        plt.tight_layout()
        save_fig(os.path.join(figures_data_path,
                    f"{country}_hazard_sector_total_damages.png"))
        plt.close()

        for dsc in ["bau","sdg"]:
            damage_data = pd.read_excel(
                            os.path.join(output_data_path,
                                "adaptation_outcomes",
                                f"{country}_aggregated_results_for_macroeconomic_analysis",
                                f"{country}_all_assets_risks_investments.xlsx"),
                            sheet_name=dsc)
            if dsc == "bau":
                damage_data = damage_data[
                                (damage_data.existing_asset_adaptation_implemented == "no") & (
                                damage_data.epoch == 2023) & (
                                damage_data.hazard != "fluvial_undefended") & (
                                damage_data.rp.isin([1,100]))
                                ]
            else:
                damage_data = damage_data[
                                    (damage_data.existing_asset_adaptation_implemented == "no") & (
                                    damage_data.epoch == 2030) & (
                                    damage_data.hazard != "fluvial_undefended") & (
                                    damage_data.rp.isin([1,100]))
                                    ]
                damage_data = damage_data[(damage_data.rcp == "ssp245") | (damage_data.hazard == "landslide")]
                
            fig, ax = plt.subplots(1,1,figsize=(8,8),dpi=500)
            all_hazards = ["landslide"] + hazards

            df = (damage_data.set_index(["subsector"]).pivot(
                                        columns="hazard"
                                        )["service_resilience_achieved_percentage"].reset_index().rename_axis(None, axis=1)).fillna(-5)

            df = df.set_index("subsector")
            data = df.to_numpy()
            v = mpl.colormaps['viridis'].resampled(256)
            newcolors = v(np.linspace(0, 1, 256))
            grey = np.array([128/256, 128/256, 128/256, 1])
            newcolors[:2, :] = grey
            newcmp = ListedColormap(newcolors)
            plt.imshow(data, interpolation='none',cmap=newcmp)
            y_labels = [s.upper() for s in df.index.values.tolist()]
            x_labels = [str(s.split("_")[0]).capitalize() for s in df.columns.values.tolist()]
            ax.tick_params(top=True, bottom=False,
                            labeltop=True, labelbottom=False)
            ax.set_xticks(np.arange(len(x_labels)), 
                                labels=x_labels,
                                fontsize=15,
                                rotation=45,
                                fontweight="bold")
            ax.set_yticks(np.arange(len(y_labels)), 
                                labels=y_labels,
                                fontsize=15,
                                fontweight="bold")
            plt.grid(None)
            # Loop over data dimensions and create text annotations.
            for i in range(len(y_labels)):
                for j in range(len(x_labels)):
                    if data[i, j] < 0.0:
                        text = ax.text(j, i, "NA",
                                   ha="center", va="center", color="w",weight='bold')
                    else:
                        text = ax.text(j, i, round(data[i, j],1),
                                       ha="center", va="center", color="w",weight='bold')

            fig.tight_layout()
            save_fig(os.path.join(figures_data_path,
                        f"{country}_hazard_sector_service_resilience_{dsc}.png"))
            plt.close()

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
