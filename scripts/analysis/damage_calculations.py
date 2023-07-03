"""Estimate direct damages to physical assets exposed to hazards

"""
import sys
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import geopandas as gpd
import numpy as np
from analysis_utils import *
from add_costs import add_costs, get_adaptation_uplifts_reductions
from tqdm import tqdm
tqdm.pandas()

epsg_caribbean = 32620

def get_damage_data(x,damage_data_path,
                    uplift_factor=0,
                    uncertainty_parameter=0):
    data = pd.read_excel(os.path.join(damage_data_path,
                        f"damage_curves_{x.sector}_{x.hazard_type}.xlsx"),
                        sheet_name=x.asset_sheet)
    if x.hazard_type == "flooding":
        x_data = data.flood_depth
    elif x.hazard_type == "TC":
        x_data = data.wind_speed
    else:
        x_data = data.landslide

    y_data = data.damage_ratio_min + uncertainty_parameter*(data.damage_ratio_max - data.damage_ratio_min)
    y_data = np.minimum(y_data*(1 + uplift_factor), 1.0)
    # y_data = np.maximum(y_data - x.vulnerability_reduction*max(y_data),0)
    y_data = y_data - x.vulnerability_reduction*max(y_data)
    return x_data.values, y_data.values

def add_exposure_dimensions(dataframe,dataframe_type="nodes",epsg=4326):
    geo_dataframe = gpd.GeoDataFrame(dataframe,
                                geometry = 'geometry',
                                crs={'init': f'epsg:{epsg}'})
    if dataframe_type == 'edges':
        geo_dataframe['exposure'] = geo_dataframe.geometry.length
        geo_dataframe['exposure_unit'] = 'm'
    elif dataframe_type == 'areas':
        geo_dataframe['exposure'] = geo_dataframe.geometry.area
        geo_dataframe['exposure_unit'] = 'm2'
    else:
        geo_dataframe['exposure'] = 1
        geo_dataframe['exposure_unit'] = 'unit'
    geo_dataframe.drop('geometry',axis=1,inplace=True)

    index_columns = [c for c in geo_dataframe.columns.values.tolist() if c != 'exposure']
    return geo_dataframe.groupby(index_columns,dropna=False)['exposure'].sum().reset_index()

def create_damage_curves(damage_data_path,
                    damage_curve_lookup_df,
                    uplift_factor=0,
                    uncertainty_parameter=0):
    damage_curve_lookup_df['x_y_data'] = damage_curve_lookup_df.progress_apply(
                                                lambda x:get_damage_data(
                                                    x,damage_data_path,
                                                    uplift_factor,
                                                    uncertainty_parameter),
                                                axis=1)
    damage_curve_lookup_df[['damage_x_data','damage_y_data']] = damage_curve_lookup_df['x_y_data'].apply(pd.Series)
    damage_curve_lookup_df.drop('x_y_data',axis=1,inplace=True)

    return damage_curve_lookup_df

def estimate_direct_damage_costs_and_units(dataframe,damage_ratio_columns,
                        cost_unit_column,damage_cost_column='damage_cost',dataframe_type="nodes"):
    if dataframe_type == "nodes":
        dataframe[damage_ratio_columns] = dataframe[damage_ratio_columns].multiply(dataframe[damage_cost_column],axis="index")
        dataframe['damage_cost_unit'] = dataframe[cost_unit_column]
    else:
        dataframe[damage_ratio_columns] = dataframe[damage_ratio_columns].multiply(dataframe[damage_cost_column]*dataframe['exposure'],axis="index")
        cost_unit = dataframe[cost_unit_column].values.tolist()[0]
        dataframe['damage_cost_unit'] = "/".join(cost_unit.split('/')[:-1])
    dataframe[damage_ratio_columns] = np.where(dataframe[damage_ratio_columns]<0,0,dataframe[damage_ratio_columns])
    return dataframe

# def modify_epoch(hazard_dataframe,baseline_year):
#     if "epoch" not in hazard_dataframe.columns.values.tolist():
#         hazard_dataframe["epoch"] = baseline_year

#     hazard_dataframe["epoch"] = hazard_dataframe["epoch"].fillna(baseline_year)
#     hazard_dataframe.loc[hazard_dataframe["epoch"] < baseline_year,"epoch"] = baseline_year
#     return hazard_dataframe

def main(config,country,hazard_name,results_folder,
        network_csv,hazard_csv,
        damage_curves_csv,
        hazard_damage_parameters_csv,
        adaptation_option,
        development_scenario,
        set_count,cost_uncertainty_parameter,damage_uncertainty_parameter):
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    baseline_year = 2023
    direct_damages_results = os.path.join(output_data_path,f"{results_folder}_{development_scenario}")
    if os.path.exists(direct_damages_results) == False:
        os.mkdir(direct_damages_results)

    hazard_asset_intersection_path = os.path.join(
                                output_data_path,
                                "hazard_asset_intersections")
    damage_curve_data_path = os.path.join(processed_data_path,
                                            "damage_curves")
    
    asset_data_details = pd.read_csv(network_csv)
    hazard_data_details = pd.read_csv(hazard_csv,encoding="latin1")
    hazard_data_details = modify_epoch(hazard_data_details,baseline_year)
    epochs = list(set(hazard_data_details["epoch"].values.tolist()))
    damage_curve_lookup = pd.read_csv(damage_curves_csv)[['sector',
                                                        'hazard_type',
                                                        'asset_name',
                                                        'asset_sheet']]
    if adaptation_option == "no_adaptation":
        damage_curve_lookup["vulnerability_reduction"] = 0
    else:
        reduction_df = get_adaptation_uplifts_reductions()
        damage_curve_lookup = pd.merge(damage_curve_lookup,
                                    reduction_df[['sector','hazard_type','asset_name','vulnerability_reduction']],
                                    how="left",
                                    on=['sector','hazard_type','asset_name']).fillna(0)

    
    hazard_attributes = pd.read_csv(hazard_damage_parameters_csv)
    flood_hazards = hazard_attributes[hazard_attributes["hazard_type"] == "flooding"]["hazard"].values.tolist()
    
    """Step 1: Get all the damage curves into a dataframe
    """
    damage_curves = []
    for idx, hazard in hazard_attributes.iterrows():
        damage_curve_df = damage_curve_lookup[damage_curve_lookup['hazard_type'] == hazard['hazard_type']]
        damage_curve_df['hazard'] = hazard['hazard']

        damage_curve_df = create_damage_curves(damage_curve_data_path,
                                                damage_curve_df,
                                                uplift_factor=hazard['uplift_factor'],
                                                uncertainty_parameter=damage_uncertainty_parameter)
        damage_curves.append(damage_curve_df)

    damage_curves = pd.concat(damage_curves,axis=0,ignore_index=True)
    del damage_curve_df

    """Step 2: Loop through the assets and estimate the damages
    """
    for asset_info in asset_data_details.itertuples():
        asset_sector = asset_info.sector
        asset_id = asset_info.asset_id_column
        asset_min_cost = asset_info.rehab_cost_min 
        asset_max_cost = asset_info.rehab_cost_max
        asset_cost_unit = asset_info.rehab_cost_unit
        hazard_damages = []
        # grd_ports_splits__deltares_storm_surge_grd__areas.geoparquet
        hazard_intersection_file = os.path.join(hazard_asset_intersection_path,
                                f"{country}_{asset_info.asset_gpkg}_splits__{hazard_name}_{country}__{asset_info.asset_layer}.geoparquet")
        if os.path.isfile(hazard_intersection_file) is True: 
            asset_df = gpd.read_file(os.path.join(processed_data_path,
                                            "infrastructure",
                                            asset_sector,
                                            f"{country}_{asset_info.asset_gpkg}.gpkg"),
                                    layer=asset_info.asset_layer)
            hazard_df = gpd.read_parquet(hazard_intersection_file)
            hazard_df = hazard_df.to_crs(epsg=epsg_caribbean)
            hazard_df = add_exposure_dimensions(hazard_df,
                                                dataframe_type=asset_info.asset_layer,
                                                epsg=epsg_caribbean)
            for hazard_info in hazard_attributes.itertuples():
                if getattr(asset_info,f"{hazard_info.hazard}_asset_damage_lookup_column") != 'none':
                    asset_hazard = getattr(asset_info,f"{hazard_info.hazard}_asset_damage_lookup_column")
                    for epoch in epochs:
                        hazard_keys = hazard_data_details[
                                                (hazard_data_details["hazard"] == hazard_info.hazard
                                                ) & (hazard_data_details["epoch"] == epoch)
                                                ]["key"].values.tolist()
                        if len(hazard_keys) > 0:
                            hazard_effect_df = hazard_df[[asset_id,'exposure','exposure_unit'] + hazard_keys]
                            damages_df = damage_curves[
                                                        (
                                                            damage_curves['sector'] == asset_sector
                                                        ) & (
                                                            damage_curves['hazard'] == hazard_info.hazard
                                                            )
                                                        ]
                            damaged_assets = list(set(damages_df['asset_name'].values.tolist()))
                            # if asset_info.asset_gpkg == "energy" and asset_info.asset_layer in ("nodes","areas"):
                            #     asset_df = add_costs(asset_df,country,
                            #                             asset_sector,asset_info.asset_gpkg,
                            #                             asset_info.asset_layer,["rehabilitation"],epoch,development_scenario)
                            asset_df = add_costs(asset_df,country,
                                                        asset_sector,asset_info.asset_gpkg,
                                                        asset_info.asset_layer,["rehabilitation"],epoch,development_scenario)
                            asset_df['damage_cost'] = asset_df[asset_min_cost] + cost_uncertainty_parameter*(
                                                            asset_df[asset_max_cost] - asset_df[asset_min_cost]
                                                            )
                            affected_assets_df = asset_df[
                                                        asset_df[asset_hazard].isin(damaged_assets)
                                                        ][[asset_id,asset_hazard,asset_cost_unit,'damage_cost']]
                            damaged_assets = list(set(affected_assets_df[asset_hazard].values.tolist()))
                            damages_df = damages_df[damages_df['asset_name'].isin(damaged_assets)]
                            affected_assets = list(set(affected_assets_df[asset_id].values.tolist()))
                            hazard_effect_df["hazard_threshold"] = hazard_info.hazard_threshold
                            if hazard_info.hazard in flood_hazards:
                                hazard_effect_df[hazard_keys] = hazard_effect_df[hazard_keys] - hazard_info.hazard_threshold
                                hazard_effect_df = hazard_effect_df[(hazard_effect_df[hazard_keys]>0).any(axis=1)]
                            else:
                                hazard_effect_df[hazard_keys] = np.where(hazard_effect_df[hazard_keys]<=hazard_info.hazard_threshold,
                                                                        0,hazard_effect_df[hazard_keys])
                                hazard_effect_df = hazard_effect_df[(hazard_effect_df[hazard_keys]>hazard_info.hazard_threshold).any(axis=1)]
                            
                            hazard_effect_df = hazard_effect_df[hazard_effect_df[asset_id].isin(affected_assets)]

                            if len(hazard_effect_df.index) == 0:
                                print (f"* No {hazard_info.hazard} intersections with {asset_info.asset_gpkg} {asset_info.asset_layer}")
                            else: 
                                hazard_effect_df = pd.merge(hazard_effect_df,affected_assets_df,how='left',on=[asset_id])
                                # print (hazard_info.key)
                                for damage_info in damages_df.itertuples():
                                    hazard_asset_effect_df = hazard_effect_df[hazard_effect_df[asset_hazard] == damage_info.asset_name]
                                    if len(hazard_asset_effect_df.index) > 0:
                                        hazard_asset_effect_df[hazard_keys] = interp1d(damage_info.damage_x_data,damage_info.damage_y_data,
                                                    fill_value=(min(damage_info.damage_y_data),max(damage_info.damage_y_data)),
                                                    bounds_error=False)(hazard_asset_effect_df[hazard_keys])
                                        hazard_asset_effect_df = estimate_direct_damage_costs_and_units(hazard_asset_effect_df,
                                                                    hazard_keys,asset_cost_unit,dataframe_type=asset_info.asset_layer)
                                        
                                        sum_dict = dict([("exposure","sum")]+[(hk,"sum") for hk in hazard_keys])
                                        hazard_asset_effect_df = hazard_asset_effect_df.groupby([asset_id,
                                                                asset_hazard,    
                                                                'exposure_unit',
                                                                'damage_cost_unit'
                                                                ],
                                                                dropna=False).agg(sum_dict).reset_index()

                                        hazard_asset_effect_df['damage_uncertainty_parameter'] = damage_uncertainty_parameter
                                        hazard_asset_effect_df['cost_uncertainty_parameter'] = cost_uncertainty_parameter
                                        hazard_damages.append(hazard_asset_effect_df)

                                    del hazard_asset_effect_df
                                del hazard_effect_df
                else:
                    print (f"* {asset_info.asset_gpkg} {asset_info.asset_layer} not affected by {hazard_info.hazard}")
        if len(hazard_damages) > 0:
            asset_damages_results = os.path.join(direct_damages_results,f"{asset_info.asset_gpkg}_{asset_info.asset_layer}")
            if os.path.exists(asset_damages_results) == False:
                os.mkdir(asset_damages_results)
            hazard_damages = pd.concat(hazard_damages,axis=0,ignore_index=True).fillna(0)
            hazard_damages.to_parquet(os.path.join(
                        asset_damages_results,
                        f"{country}_{hazard_name}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_direct_damages_parameter_set_{set_count}.parquet"),
                        index=False)

            # hazard_damages.to_csv(os.path.join(
            #             asset_damages_results,
            #             f"{country}_{hazard_name}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_direct_damages_parameter_set_{set_count}.csv"),
            #             index=False)

if __name__ == "__main__":
    CONFIG = load_config()
    try:
        country =  str(sys.argv[1])
        hazard_name = str(sys.argv[2])
        results_folder = str(sys.argv[3])
        network_csv = str(sys.argv[4])
        hazard_csv = str(sys.argv[5])
        damage_curves_csv = str(sys.argv[6])
        hazard_damage_parameters_csv = str(sys.argv[7])
        adaptation_option = str(sys.argv[8])
        development_scenario = str(sys.argv[9])
        set_count = str(sys.argv[10])
        cost_uncertainty_parameter = float(sys.argv[11])
        damage_uncertainty_parameter = float(sys.argv[12])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()

    main(CONFIG,country,hazard_name,results_folder,
        network_csv,hazard_csv,
        damage_curves_csv,
        hazard_damage_parameters_csv,
        adaptation_option,
        development_scenario,
        set_count,cost_uncertainty_parameter,damage_uncertainty_parameter)