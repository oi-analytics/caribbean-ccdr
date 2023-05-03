"""Provide the data inputs for the analysis
"""
import sys
import os
import shutil
import pandas as pd
import geopandas as gpd
import collections
from scipy import integrate
import numpy as np
import risk_and_adaptation_functions as rad
from tqdm import tqdm
tqdm.pandas()
CARIBBEAN_CRS = 32620

def main():
    config = rad.load_config()
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['output']
    # Path where the hazard-asset intersections results are stored
    intersection_path = os.path.join(outputs_path,'hazard_asset_intersection')
    results_path = os.path.join(outputs_path,'risk_and_adaptation_results')
    if os.path.exists(results_path) == False:
        os.mkdir(results_path)

    climate_idx_cols = ['hazard', 'model','rcp', 'epoch'] # The climate columns
    network_csv = os.path.join(processed_data_path,
                            "network_layers_hazard_intersections_details_0.csv")
    sector_attributes = pd.read_csv(network_csv)

    damage_curves_csv = os.path.join(processed_data_path,
                            "damage_curves",
                            "asset_damage_curve_mapping.csv")

    damage_curve_lookup = pd.read_csv(damage_curves_csv)[['sector',
                                                        'hazard_type',
                                                        'asset_name',
                                                        'asset_sheet']]

    damage_curves_path = os.path.join(processed_data_path,
                                        "damage_curves",    
                                        "damage_curves_flooding.xlsx"
                                    ) 

    # Set up problem for sensitivity analysis
    problem = {
              'num_vars': 6,
              'names': ['fragility_parameter','cost_parameter',
                        'economic_loss_parameter','duration',
                        'discount_rate','gdp_growth_fluctuate'],
              'bounds': [[0,1],[0,1],[0,1],[2,30],[4.5,6.5],[-2.0,2.0]]
              }
    
    # And create parameter values
    param_values = morris.sample(problem,
                                100, 
                                num_levels=2, 
                                optimal_trajectories=20,
                                local_optimization=True)
    param_values = np.unique(param_values, axis=0)
    """Uniform sampling of the parameters 
    """
    # linear_intervals = 20
    # param_values = list(zip(*[np.linspace(pr[0],pr[1],linear_intervals) for pr in problem['bounds']]))

    np.savetxt(os.path.join(
                output_path,"parameter_combinations.txt"),
                param_values, delimiter=','
            )

    for asset_info in sector_attributes.itertuples():
        """Columns in dataframe
            sector  
            asset_description   
            asset_gpkg  
            asset_layer 
            asset_id_column 
            asset_min_cost_column   
            asset_max_cost_column   
            asset_min_economic_loss_column  
            asset_max_economic_loss_column  
            asset_cost_unit_column  
            asset_economic_loss_unit_column 
            flooding_asset_damage_lookup_column 
            river_asset_damage_lookup_column    
            path
        """        
        asset_sector = asset_info.asset_gpkg
        asset_id = asset_info.asset_id_column
        asset_min_cost = asset_info.asset_min_cost_column 
        asset_max_cost = asset_info.asset_max_cost_column
        asset_cost_unit = asset_info.asset_cost_unit_column
        asset_min_econ_loss = asset_info.asset_min_economic_loss_column
        asset_max_econ_loss = asset_info.asset_max_economic_loss_column
        asset_loss_unit = asset_info.asset_economic_loss_unit_column
        asset_type = asset_info.flooding_asset_damage_lookup_column

        results_path = os.path.join(output_path,asset_sector)
        if os.path.exists(results_path) == False:
            os.mkdir(results_path)

        hazard_intersection_path = os.path.join(asset_path,
                    f"{asset_sector}_splits__hazard_layers__{asset_info.asset_layer}.geoparquet")
        hazard_df = gpd.read_parquet(hazard_intersection_path).fillna(0)
        hazard_df = hazard_df.to_crs(epsg=epsg_china)
        hazard_cols_original = [c for c in hazard_df.columns.values.tolist() if c in hazard_keys]
        hazard_df = rad.add_exposure_dimensions(hazard_df,
                                            dataframe_type=asset_info.asset_layer,
                                            epsg=epsg_china)
        hazard_df = hazard_df[[asset_id,'exposure','exposure_unit'] + hazard_cols_original]
        sector_data_df = gpd.read_file(os.path.join(processed_data_path,
                                        "networks",
                                        f"{asset_sector}.gpkg"),
                                        layer=asset_info.asset_layer)
        # a_cols = sector_data_df.columns.values.tolist()
        # print (a_cols)
        sector_data_df = pd.merge(hazard_df,sector_data_df,how="left",on=[asset_id])
        # print ('Columns:',[a for a in sector_data_df.columns.values if a in a_cols])
        rp_min = 0
        change = 0
        while rp_min < flood_return_periods[-1]:
            if change == 0:
                flood_protection_name = 'no_protection_rp'
                sector_data_df[flood_protection_name] = 0
                change = 1
            elif change == 1:
                flood_protection_name = asset_info.flood_protection_column
                change = 2
            else:
                sector_data_df, flood_protection_name = rad.change_flood_protection_standards(sector_data_df,
                                                                                            flood_protection_name,
                                                                                            flood_return_periods)
            rp_min = sector_data_df[flood_protection_name].min()

            pr_rps_cols = [p for p in protection_rps_cols if p[1] in hazard_cols_original]
            
            sector_data_df, hazard_cols = rad.get_hazard_magnitude(sector_data_df,
                                                        hazard_cols_original,
                                                        flood_protection_name,pr_rps_cols)
            # print ('Columns:',[a for a in sector_data_df.columns.values if a in a_cols])
            exposures = sector_data_df.copy()
            exposures[hazard_cols] = exposures["exposure"].to_numpy()[:,None]*np.where(exposures[hazard_cols]>0,1,0)
            sum_dict = dict([(hk,"sum") for hk in hazard_cols])
            exposures = exposures.groupby([asset_id,
                                        flood_protection_name,
                                        'exposure_unit'
                                        ],
                                        dropna=False).agg(sum_dict).reset_index()
            # exposures.to_csv(os.path.join(results_path,
            #                 f"{asset_sector}_{asset_info.asset_layer}_{flood_protection_name}_exposures.csv"),
            #                 index=False)
            exposures.to_parquet(os.path.join(results_path,
                            f"{asset_sector}_{asset_info.asset_layer}_{flood_protection_name}_exposures.parquet"),
                            index=False)
            del exposures

            if flood_protection_name != "no_protection_rp":
                # sector_npvs = rad.get_adaptation_benefits(sector_npvs,sector_no_protection_npvs,flood_protection_name)
                print ('* Add adaptation costs')
                sector = dict([("sector",asset_sector),
                                ("id_column",asset_id),
                                ("cost_column",asset_info.asset_mean_cost_column),
                                ("length_column",asset_info.length_column),
                                ("length_unit",asset_info.length_unit),
                                ("adaptation_criteria_column",asset_info.adaptation_criteria_column),
                                ("flood_protection_column",flood_protection_name),
                                ("cost_conversion",1.0)])
                # print ('Columns:',[a for a in sector_data_df.columns.values if a in a_cols])
                adapt_costs_df = rad.assign_adaptation_costs(sector_data_df,
                                                        sector,adaptation_data_path,
                                                        flood_protection_name)
            sector_damages_all = []
            sector_losses_all = []
            sector_risks_all = []
            sector_ead_timeseries_all = []
            sector_eael_timeseries_all = []
            sector_ead_timeseries_discounted = []
            sector_eael_timeseries_discounted = []
            sector_npv_all = []
            benefits_all = []
            param_scenario = 0
            for param in param_values:
                param_scenario += 1
                fragility_parameter = param[0]
                damage_cost_parameter = param[1]
                economic_loss_parameter = param[2]
                duration = param[3]
                discount_rate = param[4]
                gdp_growth_fluctuate = param[5]

                param_results_path = os.path.join(results_path,f"param_{param_scenario}")
                if os.path.exists(param_results_path) == False:
                    os.mkdir(param_results_path)
                # index_cols = [
                #         sector['id_column'],
                #         sector['flood_protection_column'],
                #         sector['cost_column']
                #     ] + climate_idx_cols + \
                #     list(set(sector['min_economic_loss_column'] + \
                #     sector['max_economic_loss_column'])) # The pivot columns for which all values will be grouped

                print ('* Estimate damages and losses')
                sector_damages = rad.direct_damage_estimation(sector_data_df,
                                                        [asset_id,flood_protection_name,
                                                        asset_min_econ_loss,asset_max_econ_loss],
                                                        asset_type,
                                                        asset_cost_unit,
                                                        hazard_cols,
                                                        asset_min_cost,
                                                        asset_max_cost,
                                                        asset_info.asset_layer,
                                                        damage_curves_path,
                                                        damage_curve_lookup,
                                                        damage_uncertainty_parameter=fragility_parameter,
                                                        cost_uncertainty_parameter=damage_cost_parameter)
                sector_losses = sector_damages.copy()
                sector_losses['economic_loss'] = duration*(sector_losses[asset_min_econ_loss] + economic_loss_parameter*(
                                        sector_losses[asset_max_econ_loss] - sector_losses[asset_min_econ_loss]))
                # print (sector_losses)
                sector_losses[hazard_cols] = sector_losses["economic_loss"].to_numpy()[:,None]*np.where(sector_losses[hazard_cols]>0,1,0)
                sector_losses.drop("economic_loss",axis=1,inplace=True)

                print ('* Estimate EAD and EAEL')
                # sector_risks_undefended = rad.EAD_EAEL_pivot(sector_damages,sector,index_cols,
                #                         flood_protection=None,flood_protection_name=None)
                # sector_risks = rad.EAD_EAEL_pivot(sector_damages,sector,index_cols,
                #                         flood_protection=1,flood_protection_name=flood_protection_name)
                sector_eads = rad.risk_estimations(hazard_data_details,climate_idx_cols,
                                            sector_damages,asset_id,'EAD',flood_protection_name)
                sector_eaels = rad.risk_estimations(hazard_data_details,climate_idx_cols,
                                            sector_losses,asset_id,'EAEL',flood_protection_name)
                # print (sector_eads)
                # print (sector_eaels)
                sector_risks = pd.merge(sector_eads,sector_eaels,how='left',on=[asset_id,flood_protection_name]).fillna(0)
                # del sector_eads, sector_eaels

                sector_damages['damage_cost_parameter'] = damage_cost_parameter
                sector_damages['fragility_parameter'] = fragility_parameter
                # sector_losses['fragility_parameter'] = fragility_parameter
                sector_losses['economic_loss_parameter'] = economic_loss_parameter
                sector_losses['duration'] = duration

                sector_damages.to_parquet(os.path.join(param_results_path,
                            f"{asset_sector}_{asset_info.asset_layer}_{flood_protection_name}_damages.parquet"),
                            index=False)
                # sector_damages_all.append(sector_damages)
                del sector_damages

                sector_losses.to_parquet(os.path.join(param_results_path,
                            f"{asset_sector}_{asset_info.asset_layer}_{flood_protection_name}_losses.parquet"),
                            index=False)
                # sector_losses_all.append(sector_losses)
                del sector_losses

                sector_risks['damage_cost_parameter'] = damage_cost_parameter
                sector_risks['fragility_parameter'] = fragility_parameter
                sector_risks['economic_loss_parameter'] = economic_loss_parameter
                sector_risks['duration'] = duration

                sector_risks.to_parquet(os.path.join(param_results_path,
                                    f'{asset_sector}_{asset_info.asset_layer}_{flood_protection_name}_risks.parquet'),
                                                    index=False)
                # sector_risks_all.append(sector_risks)
                del sector_risks

                # sector_risks_costs = pd.merge(sector_risks,adapt_costs,
                #                         how='left',
                #                         on=[asset_id])
                # # del sector_risks

                # # EAEL_cols = [c for c in sector_risks_costs.columns.values.tolist() if 'EAEL_' in c]
                # # for e in EAEL_cols:
                # #     sector_risks_costs[e] = duration*sector_risks_costs[e]

                # index_cols = [sector['id_column']] +  ['hazard', 'model','rcp']

                print ('* Estimate EAD and EAEL timeseries and NPV')
                gr_rates = [(y,gr_rate + gdp_growth_fluctuate) for (y,gr_rate) in growth_year_rates]

                ead_time_series,ead_time_series_discounted,ead_discounted_values = rad.estimate_time_series(hazard_data_details,
                                                                    sector_eads,
                                                                    [asset_id,flood_protection_name],
                                                                    "EAD",
                                                                    flood_protection_name,
                                                                    start_year,
                                                                    end_year,
                                                                    gr_rates,
                                                                    discount_rate)
                eael_time_series,eael_time_series_discounted,eael_discounted_values = rad.estimate_time_series(hazard_data_details,
                                                                    sector_eaels,
                                                                    [asset_id,flood_protection_name],
                                                                    "EAEL",
                                                                    flood_protection_name,
                                                                    start_year,
                                                                    end_year,
                                                                    gr_rates,
                                                                    discount_rate)

                ead_time_series['damage_cost_parameter'] = damage_cost_parameter
                ead_time_series['fragility_parameter'] = fragility_parameter

                # eael_time_series['fragility_parameter'] = fragility_parameter
                eael_time_series['economic_loss_parameter'] = economic_loss_parameter
                eael_time_series['duration'] = duration
                eael_time_series['growth_rate'] = gdp_growth_fluctuate

                ead_time_series_discounted['damage_cost_parameter'] = damage_cost_parameter
                ead_time_series_discounted['fragility_parameter'] = fragility_parameter
                ead_time_series_discounted['discount_rate'] = discount_rate
                
                # eael_time_series_discounted['fragility_parameter'] = fragility_parameter
                eael_time_series_discounted['economic_loss_parameter'] = economic_loss_parameter
                eael_time_series_discounted['duration'] = duration
                eael_time_series_discounted['growth_rate'] = gdp_growth_fluctuate
                eael_time_series_discounted['discount_rate'] = discount_rate

                ead_time_series.columns = ead_time_series.columns.astype(str)
                ead_time_series.to_parquet(os.path.join(param_results_path,
                            f'{asset_sector}_{asset_info.asset_layer}_{flood_protection_name}_ead_timeseries.parquet'),
                                                    index=False)
                eael_time_series.columns = eael_time_series.columns.astype(str)
                eael_time_series.to_parquet(os.path.join(param_results_path,
                            f'{asset_sector}_{asset_info.asset_layer}_{flood_protection_name}_eael_timeseries.parquet'),
                                                        index=False)
                ead_time_series_discounted.columns = ead_time_series_discounted.columns.astype(str)
                ead_time_series_discounted.to_parquet(os.path.join(param_results_path,
                            f'{asset_sector}_{asset_info.asset_layer}_{flood_protection_name}_ead_timeseries_discounted.parquet'),
                                                        index=False)
                eael_time_series_discounted.columns = eael_time_series_discounted.columns.astype(str)
                eael_time_series_discounted.to_parquet(os.path.join(param_results_path,
                            f'{asset_sector}_{asset_info.asset_layer}_{flood_protection_name}_eael_timeseries_discounted.parquet'),
                                                        index=False)
                # sector_ead_timeseries_all.append(ead_time_series)
                del ead_time_series

                # sector_eael_timeseries_all.append(eael_time_series)
                del eael_time_series

                # sector_ead_timeseries_discounted.append(ead_time_series_discounted)
                del ead_time_series_discounted

                # sector_eael_timeseries_discounted.append(eael_time_series_discounted)
                del eael_time_series_discounted

                sector_npvs = pd.merge(ead_discounted_values,eael_discounted_values,how="left",
                                        on=[asset_id,flood_protection_name,"hazard","model","rcp"])
                sector_npvs['damage_cost_parameter'] = damage_cost_parameter
                sector_npvs['fragility_parameter'] = fragility_parameter
                sector_npvs['economic_loss_parameter'] = economic_loss_parameter
                sector_npvs['duration'] = duration
                sector_npvs['growth_rate'] = gdp_growth_fluctuate
                sector_npvs['discount_rate'] = discount_rate

                if flood_protection_name != "no_protection_rp":
                    # sector_npvs = rad.get_adaptation_benefits(sector_npvs,sector_no_protection_npvs,flood_protection_name)
                    # print ('* Add adaptation costs')
                    # sector = dict([("sector",asset_sector),
                    #                 ("id_column",asset_id),
                    #                 ("cost_column",asset_info.asset_mean_cost_column),
                    #                 ("length_column",asset_info.length_column),
                    #                 ("length_unit",asset_info.length_unit),
                    #                 ("adaptation_criteria_column",asset_info.adaptation_criteria_column),
                    #                 ("flood_protection_column",flood_protection_name),
                    #                 ("cost_conversion",1.0)])
                    # # print ('Columns:',[a for a in sector_data_df.columns.values if a in a_cols])
                    # adapt_costs = rad.assign_adaptation_costs(sector_data_df,
                    #                                         sector,adaptation_data_path,
                    #                                         flood_protection_name)
                    adapt_costs, adapt_cost_columns = rad.adaptation_costs_npvs(adapt_costs_df.copy(),
                                                        asset_id,
                                                        flood_protection_name,
                                                        "mean",
                                                        discount_rate=discount_rate,
                                                        start_year=start_year,end_year=end_year
                                                        )
                    sector_npvs = pd.merge(sector_npvs,adapt_costs,how="left",on=[asset_id,flood_protection_name])
                # else:
                #     sector_no_protection_npvs = sector_npvs.copy()
                #     sector_no_protection_npvs.drop("no_protection_rp",axis=1,inplace=True)

                sector_npvs.to_parquet(os.path.join(param_results_path,
                                    f'{asset_sector}_{asset_info.asset_layer}_{flood_protection_name}_npvs.parquet'),
                                                    index=False)
                # sector_npv_all.append(sector_npvs)
                del sector_npvs

            # Merge all the results now
            for end_string in ['damages','losses',
                            'risks','npvs','ead_timeseries',
                            'eael_timeseries','ead_timeseries_discounted',
                            'eael_timeseries_discounted']:
                dfs = [pd.read_parquet(os.path.join(results_path,f"param_{i+1}",
                            f'{asset_sector}_{asset_info.asset_layer}_{flood_protection_name}_{end_string}.parquet')) for i in range(len(param_values))]

                combined_df = pd.concat(dfs,
                                        axis=0,
                                        sort='False',
                                        ignore_index=True).fillna(0)
                combined_df.columns = combined_df.columns.astype(str)
                if end_string == 'npvs':
                    if flood_protection_name == "no_protection_rp": 
                        sector_no_protection_npvs = combined_df.copy()
                        sector_no_protection_npvs.drop("no_protection_rp",axis=1,inplace=True)
                    else:
                        combined_df = rad.get_adaptation_benefits(combined_df,
                                                                    sector_no_protection_npvs,
                                                                    [flood_protection_name] + adapt_cost_columns)
                
                combined_df.to_parquet(os.path.join(results_path,
                                f"{asset_sector}_{asset_info.asset_layer}_{flood_protection_name}_{end_string}.parquet"),
                                index=False)
                del combined_df
            
            # Delete all the sub-folders
            for i in range(len(param_values)):
                # os.remove(os.path.join(results_path,f"param_{i+1}"))
                shutil.rmtree(os.path.join(results_path,f"param_{i+1}"))
            
if __name__ == "__main__":
    main()