# -*- coding: utf-8 -*-
"""Estimate costs and benefits under fixed parameters
varying the 
    cost components, 
    durations of disruptions, 
    GDP growth rates
"""
import os
import sys
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import math
from tqdm import tqdm
tqdm.pandas()

def load_config():
    """Read config.json
    """
    config_path = os.path.join(os.path.dirname(__file__), '..','..','config.json')
    with open(config_path, 'r') as config_fh:
        config = json.load(config_fh)
    return config

def change_flood_protection_standards(dataframe,flood_protection_column,flood_return_periods):
    given_flood_protection = sorted(list(set(dataframe[flood_protection_column].values.tolist())))
    if len(given_flood_protection) == 1:
        new_protection_rp = [flood_return_periods[f+1] for f in range(len(flood_return_periods)-1) if flood_return_periods[f] <= given_flood_protection[0] < flood_return_periods[f+1]]
        if new_protection_rp:
            flood_protection_column = '{}_to_{}_year_protection'.format(int(given_flood_protection[0]),int(new_protection_rp[0]))
            dataframe[flood_protection_column] = new_protection_rp[0]
            return dataframe, flood_protection_column
        else:
            print ('* Flood return periods should exceed design standards - Cannot upgrade further')
    else:
        new_protection_column = '{}_to_{}_year_protection'.format(int(given_flood_protection[0]),int(given_flood_protection[1]))
        dataframe[new_protection_column] = dataframe[flood_protection_column]
        dataframe.loc[dataframe[new_protection_column] == given_flood_protection[0],new_protection_column] = given_flood_protection[1]
        flood_protection_column = new_protection_column
        return dataframe, flood_protection_column 

# getting Product
def prod(val) :
    res = 1
    for ele in val:
        res *= ele
    return res

def calculate_discounting_arrays(discount_rate=4.5, growth_rate=5.0,
                                start_year=2020,end_year=2050,
                                maintain_period=4):
    """Set discount rates for yearly and period maintenance costs

    Parameters
    ----------
    discount_rate
        yearly discount rate
    growth_rate
        yearly growth rate

    Returns
    -------
    discount_rate_norm
        discount rates to be used for the costs
    discount_rate_growth
        discount rates to be used for the losses
    min_main_dr
        discount rates for 4-year periodic maintenance
    max_main_dr
        discount rates for 8-year periodic maintenance

    """
    discount_rates = []
    growth_rates = []

    for year in range(start_year,end_year+1):
        discount_rates.append(
            1.0/math.pow(1.0 + 1.0*discount_rate/100.0, year - start_year))

    if isinstance(growth_rate, float):
        for year in range(start_year,end_year+1):
            growth_rates.append(
                1.0*math.pow(1.0 + 1.0*growth_rate/100.0, year - start_year))
    else:
        for i, (year,rate) in enumerate(growth_rate):
            if year > start_year:
                growth_rates.append(prod([1 + v[1]/100.0 for v in growth_rate[:i]]))
            else:
                growth_rates.append(1)  

    maintain_years = np.arange(start_year, end_year+1,maintain_period)
    maintain_rates = []
    for year in maintain_years[1:]:
        maintain_rates.append(1.0 / math.pow(1.0 + 1.0*discount_rate/100.0, year - start_year))

    return np.array(discount_rates), np.array(growth_rates), np.array(maintain_rates)


def calculate_growth_rate(growth_rate=5.0,
                                start_year=2020,end_year=2050,
                        ):
    """Set discount rates for yearly and period maintenance costs

    Parameters
    ----------
    discount_rate
        yearly discount rate
    growth_rate
        yearly growth rate

    Returns
    -------
    discount_rate_norm
        discount rates to be used for the costs
    discount_rate_growth
        discount rates to be used for the losses
    min_main_dr
        discount rates for 4-year periodic maintenance
    max_main_dr
        discount rates for 8-year periodic maintenance

    """
    growth_rates = []

    if isinstance(growth_rate, float):
        for year in range(start_year,end_year+1):
            growth_rates.append(
                1.0*math.pow(1.0 + 1.0*growth_rate/100.0, year - start_year))
    else:
        for i, (year,rate) in enumerate(growth_rate):
            if year > start_year:
                growth_rates.append(prod([1 + v[1]/100.0 for v in growth_rate[:i]]))
            else:
                growth_rates.append(1)  

    return np.array(growth_rates)


def calculate_discounting_rate(discount_rate=4.5,
                                start_year=2020,end_year=2050,
                                maintain_period=4,skip_year_one=False):
    """Set discount rates for yearly and period maintenance costs

    Parameters
    ----------
    discount_rate
        yearly discount rate
    growth_rate
        yearly growth rate

    Returns
    -------
    discount_rate_norm
        discount rates to be used for the costs
    discount_rate_growth
        discount rates to be used for the losses
    min_main_dr
        discount rates for 4-year periodic maintenance
    max_main_dr
        discount rates for 8-year periodic maintenance

    """
    discount_rates = []
    maintain_years = np.arange(start_year+1, end_year+1,maintain_period)
    for year in range(start_year,end_year+1):
        if year in maintain_years:
            discount_rates.append(
                1.0/math.pow(1.0 + 1.0*discount_rate/100.0, year - start_year))
        else:
            if skip_year_one is True:
                discount_rates.append(0)
            else:
                discount_rates.append(1)

    return np.array(discount_rates)


def get_adaptation_options_costs(asset_dictionary,data_path,
                               discount_rate=4.5,start_year=2016,end_year=2050,
                                maintain_period=3,percentage_conversion_factor=1):

    print('* {} started!'.format(asset_dictionary['sector']))

    # load cost file
    print ('* Get adaptation costs')
    adapt = pd.read_csv(os.path.join(data_path,'risk_results','{}_risks.csv'.format(asset_dictionary['sector'])))
    if asset_dictionary['length_column'] not in (None,"None"):
        adapt['ini_investment'] = (1.0*asset_dictionary['cost_conversion']/asset_dictionary[
                                'length_unit'
                                ])*percentage_conversion_factor*asset_dictionary[
                                'investment_percent']*adapt[asset_dictionary[
                                'length_column']]*adapt[asset_dictionary['cost_column']]
        adapt['routine_investment'] = (1.0*asset_dictionary[
                                    'cost_conversion']/asset_dictionary[
                                    'length_unit'])*percentage_conversion_factor*asset_dictionary[
                                    'routine_maintenance_percent']*adapt[asset_dictionary[
                                    'length_column']]*adapt[asset_dictionary['cost_column']]
        adapt['periodic_investment'] = (1.0*asset_dictionary[
                                        'cost_conversion']/asset_dictionary[
                                        'length_unit'])*percentage_conversion_factor*asset_dictionary[
                                        'periodic_maintenance_percent']*adapt[asset_dictionary[
                                        'length_column']]*adapt[asset_dictionary['cost_column']]
    
    else:
        adapt['ini_investment'] = percentage_conversion_factor*asset_dictionary[
                                    'cost_conversion']*asset_dictionary[
                                    'investment_percent']*adapt[asset_dictionary['cost_column']]
        adapt['routine_investment'] = percentage_conversion_factor*asset_dictionary[
                                    'cost_conversion']*asset_dictionary[
                                    'routine_maintenance_percent']*adapt[asset_dictionary['cost_column']]
        adapt['periodic_investment'] = percentage_conversion_factor*asset_dictionary[
                                    'cost_conversion']*asset_dictionary[
                                    'periodic_maintenance_percent']*adapt[asset_dictionary['cost_column']]

    # adapt['maintenance_cost'] = 0.01*asset_dictionary['maintenance_percent']*adapt['ini_investment']

    print ('* Get discount ratios')
    dr_norm, dr_growth, main_dr = calculate_discounting_arrays(
        discount_rate, 0.0, start_year,end_year,maintain_period)

    adapt['tot_maintenance_cost_npv'] = sum(dr_norm)*adapt['routine_investment'] + sum(main_dr)*adapt['periodic_investment']
    adapt['tot_adap_cost_npv'] = adapt['ini_investment'] + adapt['tot_maintenance_cost_npv']

    if asset_dictionary['length_column'] not in (None,"None"):
        adapt['tot_adap_cost_npv_per_km'] = adapt['tot_adap_cost_npv']/(adapt[asset_dictionary['length_column']]/asset_dictionary['length_unit'])

    return adapt

def line_length_km(line, ellipsoid='WGS-84'):
    """Length of a line in meters, given in geographic coordinates.

    Adapted from https://gis.stackexchange.com/questions/4022/looking-for-a-pythonic-way-to-calculate-the-length-of-a-wkt-linestring#answer-115285

    Args:
        line: a shapely LineString object with WGS-84 coordinates.

        ellipsoid: string name of an ellipsoid that `geopy` understands (see http://geopy.readthedocs.io/en/latest/#module-geopy.distance).

    Returns:
        Length of line in kilometers.
    """
    if line.geometryType() == 'MultiLineString':
        return sum(line_length_km(segment) for segment in line)

    return sum(
        distance.distance(tuple(reversed(a)), tuple(reversed(b)),ellipsoid=ellipsoid).km
        for a, b in pairwise(line.coords)
    )

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

def interpolate_rp_factor(df,protection_standard_column):
    return (
        (np.log(df.rp) - np.log(df.rp_l))
        / (np.log(df.rp_u) - np.log(df.rp_l)))


def interpolate_depth_df(df):
    depth = df.depth_l + (
        (df.depth_u - df.depth_l)
        * df.rp_factor)

    return depth

def interpolate_design_depths(hazard_exposure_df,
                            protection_standard_column,
                            return_periods_and_columns
                            ):
    
    return_periods_and_columns = sorted(return_periods_and_columns,key=lambda x: x[0])
    RPS,return_period_columns = zip(*return_periods_and_columns)
    rp_max = RPS[-1]
    rp_min = RPS[0]
    RPS = np.array([1e-3] + list(RPS) + [1e6])

    hazard_exposure_df['rp'] = hazard_exposure_df[protection_standard_column]
    hazard_exposure_df.loc[hazard_exposure_df.rp <= rp_min,'rp'] = rp_min
    hazard_exposure_df.loc[hazard_exposure_df.rp >= rp_max,'rp'] = rp_max


    bin_index = np.searchsorted(RPS, hazard_exposure_df.rp, side='left')
    hazard_exposure_df['bin_index'] = bin_index
    hazard_exposure_df['rp_l'] = RPS[bin_index-1]
    hazard_exposure_df['rp_u'] = RPS[bin_index]
    hazard_exposure_df['rp_factor'] = interpolate_rp_factor(hazard_exposure_df,protection_standard_column)
    # hazard_exposure_df is now a dataframe with added columns:
    # rp, bin_index, rp_l, rp_u, rp_factor
    depths = [0] + [hazard_exposure_df[c] for c in return_period_columns]

    hazard_exposure_df['depth_l'] = np.choose(hazard_exposure_df.bin_index - 1, depths)
    hazard_exposure_df['depth_u'] = np.choose(hazard_exposure_df.bin_index, depths)
    hazard_exposure_df['hazard_threshold'] = interpolate_depth_df(hazard_exposure_df)
    hazard_exposure_df.loc[hazard_exposure_df.hazard_threshold <= 0, 'hazard_threshold'] = 0
    hazard_exposure_df.drop(['bin_index','rp','rp_l','rp_u','rp_factor','depth_l','depth_u'],axis=1,inplace=True)

    return hazard_exposure_df

def get_hazard_magnitude(hazard_effect_df,hazard_columns,asset_protection_standard,protection_rps_cols):
    hazard_effect_df = interpolate_design_depths(hazard_effect_df,
                                                asset_protection_standard,
                                                protection_rps_cols
                                                )
    hazard_columns_mod = [f"{c}_mod" for c in hazard_columns]
    hazard_effect_df[hazard_columns_mod] = hazard_effect_df[hazard_columns].sub(hazard_effect_df.hazard_threshold,axis=0)
    # hazard_effect_df = hazard_effect_df[(hazard_effect_df[hazard_columns_mod]>0).any(axis=1)]
    
    return hazard_effect_df, hazard_columns_mod

def get_damage_data(x,damage_data_path,
                    uplift_factor=0,
                    uncertainty_parameter=0):
    data = pd.read_excel(os.path.join(damage_data_path),
                        sheet_name=x.asset_sheet)
    if x.hazard_type == 'flooding':
        x_data = data.flood_depth
    else:
        x_data = data.wind_speed

    y_data = data.min_damage_ratio + uncertainty_parameter*(data.max_damage_ratio - data.min_damage_ratio)
    y_data = np.minimum(y_data*(1 + uplift_factor), 1.0)

    return x_data.values, y_data.values

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
    
    return dataframe

def direct_damage_estimation(hazard_effect_df,
                            index_columns,
                            asset_type,
                            asset_cost_unit,
                            hazard_keys,
                            asset_min_cost,
                            asset_max_cost,
                            asset_layer,
                            damage_data_path,
                            damage_curve_lookup_df,
                            uplift_factor=0,
                            damage_uncertainty_parameter=0,
                            cost_uncertainty_parameter=0):
    hazard_damages = []
    damage_df = create_damage_curves(damage_data_path,
                                    damage_curve_lookup_df,
                                    uplift_factor=0,
                                    uncertainty_parameter=damage_uncertainty_parameter)
    hazard_effect_df['damage_cost'] = hazard_effect_df[asset_min_cost] + cost_uncertainty_parameter*(
                                        hazard_effect_df[asset_max_cost] - hazard_effect_df[asset_min_cost])
    for damage_info in damage_df.itertuples():
        hazard_asset_effect_df = hazard_effect_df[hazard_effect_df[asset_type] == damage_info.asset_name]
        if len(hazard_asset_effect_df.index) > 0:
            hazard_asset_effect_df[hazard_keys] = interp1d(damage_info.damage_x_data,damage_info.damage_y_data,
                        fill_value=(min(damage_info.damage_y_data),max(damage_info.damage_y_data)),
                        bounds_error=False)(hazard_asset_effect_df[hazard_keys])
            hazard_asset_effect_df = estimate_direct_damage_costs_and_units(hazard_asset_effect_df,
                                        hazard_keys,asset_cost_unit,dataframe_type=asset_layer)
            
            sum_dict = dict([(hk,"sum") for hk in hazard_keys])
            hazard_asset_effect_df = hazard_asset_effect_df.groupby(index_columns + [
                                    'exposure_unit',
                                    'damage_cost_unit'
                                    ],
                                    dropna=False).agg(sum_dict).reset_index()

            hazard_asset_effect_df['fragility_parameter'] = damage_uncertainty_parameter
            hazard_asset_effect_df['damage_cost_parameter'] = cost_uncertainty_parameter
            hazard_damages.append(hazard_asset_effect_df)

        del hazard_asset_effect_df
    del hazard_effect_df

    hazard_damages = pd.concat(hazard_damages,axis=0,ignore_index=True).fillna(0)
    return hazard_damages


def risks(dataframe,index_columns,probabilities,
            expected_risk_column,
            flood_protection_period=0,flood_protection_name=None):
    
    """
    Organise the dataframe to pivot with respect to index columns
    Find the expected risks
    """
    if flood_protection_name is None and flood_protection_period == 0:
        # When there is no flood protection at all
        expected_risk_column = f"{expected_risk_column}_undefended"
        probability_columns = [str(p) for p in probabilities]
        
    elif flood_protection_period > 0:
        if flood_protection_name is None:
            expected_risk_column = f"{expected_risk_column}_{flood_protection_period}_year_protection"
        else:
            expected_risk_column = f"{expected_risk_column}_{flood_protection_name}"
        
        probabilities = [pr for pr in probabilities if pr <= 1.0/flood_protection_period]
        probability_columns = [str(p) for p in probabilities]
    else:
        # When there is no flood protection at all
        expected_risk_column = f"{expected_risk_column}_{flood_protection_name}"
        probability_columns = [str(p) for p in probabilities]
        
    dataframe.columns = dataframe.columns.astype(str)
    # dataframe[expected_risk_column] = list(integrate.trapz(dataframe[probability_columns].to_numpy(),
    #                                         np.array([probabilities*len(dataframe.index)]).reshape(dataframe[probability_columns].shape)))

    dataframe[expected_risk_column] = list(integrate.simpson(dataframe[probability_columns].to_numpy(),
                                            np.array([probabilities*len(dataframe.index)]).reshape(dataframe[probability_columns].shape)))
    
    # dataframe = dataframe[index_columns + [expected_risk_column]].set_index(index_cols)
    return dataframe[index_columns + [expected_risk_column]].set_index(index_columns)

def risk_estimations(hazard_data_details,hazard_index_columns,hazard_dataframe,asset_id,damage_type,flood_protection_name):
    hazard_data_details = hazard_data_details.set_index(hazard_index_columns)
    haz_index_vals = list(set(hazard_data_details.index.values.tolist()))
    expected_damages = []
    for hc in haz_index_vals:
        haz_df = hazard_data_details[hazard_data_details.index == hc]
        haz_cols, haz_rps = map(list,list(zip(*sorted(
                                    list(zip(haz_df.key.values.tolist(),
                                    haz_df.rp.values.tolist()
                                    )),key=lambda x:x[-1],reverse=True))))
        
        haz_cols = [f"{c}_mod" for c in haz_cols]
        haz_prob = [1.0/rp for rp in haz_rps]
        damages = hazard_dataframe[[asset_id,flood_protection_name] + haz_cols] 
        damages.columns = [asset_id,flood_protection_name] + haz_prob
        if min(haz_prob) > 0:
            damages[0] = damages[min(haz_prob)]
            haz_prob = [0] + haz_prob
        if max(haz_prob) < 1:
            damages[1] = damages[max(haz_prob)]
            haz_prob += [1]
        hz_st = '_'.join([str(h_c) for h_c in hc])
        damage_col = f"{damage_type}_{hz_st}"
        expected_damage_df = risks(damages,[asset_id, flood_protection_name],haz_prob,
                                damage_col,
                                flood_protection_name=flood_protection_name 
                                )
        # print (expected_damage_df)
        expected_damages.append(expected_damage_df)
        del expected_damage_df

    expected_damages = pd.concat(expected_damages,axis=1)
    return expected_damages.reset_index()

def extract_growth_rate_info(growth_rates,time_column,rate_column,start_year=2000,end_year=2100):
    growth_year_rates = []
    growth_rates_times = list(sorted(growth_rates[time_column].values.tolist()))
    # And create parameter values
    for y in range(start_year,end_year+1):
        if y in growth_rates_times:
            growth_year_rates.append((y,growth_rates.loc[growth_rates[time_column] == y,rate_column].values[0]))
        elif y < growth_rates_times[0]:
            growth_year_rates.append((y,growth_rates.loc[growth_rates[time_column] == growth_rates_times[0],rate_column].values[0]))
        elif y > growth_rates_times[-1]:
            growth_year_rates.append((y,growth_rates.loc[growth_rates[time_column] == growth_rates_times[-1],rate_column].values[0])) 

    return growth_year_rates

def calculate_growth_rate_factor(growth_rate=5.0,
                                start_year=2020,end_year=2050,
                        ):
    """Set discount rates for yearly and period maintenance costs

    Parameters
    ----------
    discount_rate
        yearly discount rate
    growth_rate
        yearly growth rate

    Returns
    -------
    discount_rate_norm
        discount rates to be used for the costs
    discount_rate_growth
        discount rates to be used for the losses
    min_main_dr
        discount rates for 4-year periodic maintenance
    max_main_dr
        discount rates for 8-year periodic maintenance

    """
    growth_rates = []

    if isinstance(growth_rate, float):
        for year in range(start_year,end_year+1):
            growth_rates.append(
                1.0*math.pow(1.0 + 1.0*growth_rate/100.0, year - start_year))
    else:
        for i, (year,rate) in enumerate(growth_rate):
            if year > start_year:
                growth_rates.append(prod([1 + v[1]/100.0 for v in growth_rate[:i]]))
            else:
                growth_rates.append(1)  

    return np.array(growth_rates)

def calculate_discounting_rate_factor(discount_rate=4.5,
                                start_year=2020,end_year=2050,
                                maintain_period=4,skip_year_one=False):
    """Set discount rates for yearly and period maintenance costs

    Parameters
    ----------
    discount_rate
        yearly discount rate
    growth_rate
        yearly growth rate

    Returns
    -------
    discount_rate_norm
        discount rates to be used for the costs
    discount_rate_growth
        discount rates to be used for the losses
    min_main_dr
        discount rates for 4-year periodic maintenance
    max_main_dr
        discount rates for 8-year periodic maintenance

    """
    discount_rates = []
    maintain_years = np.arange(start_year+1, end_year+1,maintain_period)
    for year in range(start_year,end_year+1):
        if year in maintain_years:
            discount_rates.append(
                1.0/math.pow(1.0 + 1.0*discount_rate/100.0, year - start_year))
        else:
            if skip_year_one is True:
                discount_rates.append(0)
            else:
                discount_rates.append(1)

    return np.array(discount_rates)

def estimate_time_series(hazard_data_details,summarised_damages,
                        index_cols,
                        risk_type,val_type,
                        baseline_year,projection_end_year,
                        growth_rates,discounting_rate):
    
    hazard_data_details.epoch = hazard_data_details.epoch.astype(int)
    years = sorted(list(set(hazard_data_details.epoch.values.tolist())))
    
    start_year = years[0]
    end_year = years[-1]
    
    if start_year < baseline_year:
        # summarised_damages.loc[summarised_damages.epoch == start_year,"epoch"] = baseline_year
        start_year = baseline_year
    if end_year < projection_end_year:
        end_year = projection_end_year

    dsc_rate = calculate_discounting_rate_factor(discount_rate=discounting_rate,
                                    start_year=start_year,end_year=end_year,maintain_period=1)
    timeseries = np.arange(start_year,end_year+1,1)
    # hazard_data_details = hazard_data_details[hazard_index_columns].drop_duplicates(subset=hazard_index_columns,keep="first")
    # hazard_baseline = hazard_data_details[hazard_data_details['rcp'] == 'baseline']

    
    # haz_index_vals = list(set(hazard_data_details.index.values.tolist()))
    # baseline_index_vals = [hc for hc in haz_index_vals if hc[1] == "baseline"]
    # hz_st = '_'.join([str(h_c) for h_c in hc])
    # damage_col = f"{risk_type}_{hz_st}"


    hazard_model_rcp = list(set(zip(hazard_data_details.hazard.values.tolist(),
                            hazard_data_details.model.values.tolist(),
                            hazard_data_details.rcp.values.tolist())))
    hazard_baseline_rcp = [hz_m_rcp for hz_m_rcp in hazard_model_rcp if hz_m_rcp[-1] == "baseline"]
    hazard_model_rcp = [hz_m_rcp for hz_m_rcp in hazard_model_rcp if hz_m_rcp[-1] != "baseline"]
    
    defence_column = [c for c in summarised_damages.columns.values.tolist() if f"{risk_type}_" in c and f"_{val_type}" in c][0]
    damages_time_series = []
    damages_time_series_discounted = []
    discounted_values = []
    for ix, (haz,mod,rcp) in enumerate(hazard_model_rcp):
        haz_rcp_damages = hazard_data_details[hazard_data_details.hazard == haz]
        years = sorted(list(set(haz_rcp_damages.epoch.values.tolist())))
        base_rcp = [hz_m_rcp for hz_m_rcp in hazard_baseline_rcp if hz_m_rcp[0] == haz][0]
        damage_cols = [
                        f"{risk_type}_{base_rcp[0]}_{base_rcp[1]}_{base_rcp[2]}_{years[0]}_{val_type}"
                        ] + [
                        f"{risk_type}_{haz}_{mod}_{rcp}_{yr}_{val_type}" for yr in years[1:]
                    ]
        df = summarised_damages[index_cols + damage_cols]
        df.columns = index_cols + [start_year] + years[1:]
        df["hazard"] = haz
        df["model"] = mod
        df["rcp"] = rcp
        years = [start_year] + years[1:]
        series = np.array([list(timeseries)*len(df.index)]).reshape(len(df.index),len(timeseries))
        df[series[0]] = interp1d(years,df[years],fill_value="extrapolate",bounds_error=False)(series[0])
        df[series[0]] = df[series[0]].clip(lower=0.0)
        if risk_type == "EAEL":
            # gr_rates = extract_growth_rate_info(growth_rates,"year",f"gdp_{val_type}",start_year=start_year,end_year=end_year)
            gr_rates = calculate_growth_rate_factor(growth_rates,start_year,end_year)
            df[series[0]] = np.multiply(df[series[0]],gr_rates)
        damages_time_series.append(df)
        df_copy = df.copy()
        df_copy[series[0]] = np.multiply(df_copy[series[0]],dsc_rate)
        damages_time_series_discounted.append(df_copy)
        # df_copy[f"{haz}__rcp_{rcp}__{risk_type}"] = df_copy[series[0]].sum(axis=1)
        df_copy[f"{risk_type}_{haz}"] = df_copy[series[0]].sum(axis=1)
        # discounted_values.append(df_copy[index_cols + ["model","rcp"] + [f"{haz}__rcp_{rcp}__{risk_type}_{val_type}"]])
        # discounted_values.append(df_copy[index_cols + ["hazard","model","rcp"] + [f"{haz}__rcp_{rcp}__{risk_type}"]])
        discounted_values.append(df_copy[index_cols + ["hazard","model","rcp"] + [f"{risk_type}_{haz}"]])
        del df, df_copy

    damages_time_series = pd.concat(damages_time_series,axis=0,ignore_index=False)
    damages_time_series_discounted = pd.concat(damages_time_series_discounted,axis=0,ignore_index=False)
    index_columns = [c for c in damages_time_series.columns.values.tolist() if c not in timeseries]

    discounted_values = pd.concat(discounted_values,axis=0,ignore_index=False)

    return (damages_time_series[index_columns + list(timeseries)],
            damages_time_series_discounted[index_columns + list(timeseries)],
            discounted_values)

def linear_scaling_risk_pivot(v,risk_years,start_year,end_year,discount_rates,growth_rates,apply_growth_rates):
    risk_values_years = sorted([(getattr(v,str(p)),p) for p in risk_years],key=lambda x: x[1])
    risk_all_years = []
    for rvy in range(len(risk_values_years)-1):
        st_series = risk_values_years[rvy]
        end_series = risk_values_years[rvy+1]
        risk_all_years += list(np.linspace(st_series[0],end_series[0],num=end_series[1]-st_series[1],endpoint=False))

    risk_times = np.array(risk_all_years + [risk_values_years[-1][0]])
    risk_times_scaled = risk_times.copy()
    if apply_growth_rates is True:
        risk_times = np.multiply(risk_times,growth_rates)
        risk_times_growth = risk_times.copy()
        risk_times = np.multiply(risk_times,discount_rates)
        return risk_times_scaled,risk_times_growth,risk_times
    else:
        risk_times = np.multiply(risk_times,discount_rates)
        return risk_times_scaled,risk_times

def risk_timeseries_climate_scenarios_pivot(dataframe,index_columns,id_column,time_column,baseline_year,
            risk_column,climate_scenario_column,
            start_year,end_year,discount_rates,growth_rates,apply_growth_rates=False):
    
    baseline_dataframe = dataframe[dataframe[time_column] == baseline_year]
    baseline_dataframe.rename(columns={risk_column:str(baseline_year)},inplace=True)
    
    times = sorted(list(set(dataframe[time_column].values.tolist())))
    timeseries = (dataframe[dataframe[time_column] != baseline_year].set_index(index_columns).pivot(
                                    columns=time_column
                                    )[risk_column].reset_index().rename_axis(None, axis=1)).fillna(0)
    timeseries.columns = timeseries.columns.astype(str)

    timeseries = pd.merge(timeseries,baseline_dataframe[[id_column,str(baseline_year)]],
                    how='left',
                    on=[id_column]).fillna(0)
    
    timeseries[f'{risk_column}_timeseries'] = timeseries.progress_apply(lambda x: linear_scaling_risk_pivot(x,times,
                                                        start_year,
                                                        end_year,
                                                        discount_rates,
                                                        growth_rates,
                                                        apply_growth_rates),axis=1)
    if apply_growth_rates is True:
        timeseries[[f"{risk_column}_timeseries_scaled",
                    f"{risk_column}_timeseries_growth",
                    f"{risk_column}_timeseries_npv"]] = timeseries[f'{risk_column}_timeseries'].apply(pd.Series)
        year_columns = [f"{risk_column}_timeseries_scaled",
                        f"{risk_column}_timeseries_growth",
                        f"{risk_column}_timeseries_npv"]
    else:
        timeseries[[f"{risk_column}_timeseries_scaled",
                    f"{risk_column}_timeseries_npv"]] = timeseries[f'{risk_column}_timeseries'].apply(pd.Series)
        year_columns = [f"{risk_column}_timeseries_scaled",
                        f"{risk_column}_timeseries_npv"]

    timeseries[f'total_{risk_column}_npv'] = timeseries.progress_apply(lambda x: sum(x[
                                                                    f'{risk_column}_timeseries_npv'
                                                                    ]),axis=1)
    # year_columns = ['{}_{}'.format(avoided_risks_column_name,y) for y in range(start_year,end_year+1)]
    year_columns += [f'total_{risk_column}_npv']        

    # return timeseries[index_columns + year_columns], year_columns
    return timeseries[index_columns + year_columns]

def get_maintenance_costs(routine_cost,periodic_cost,discount_rate,start_year,end_year,period_freq):
    routine_rates = calculate_discounting_rate_factor(discount_rate,start_year,end_year,maintain_period=1,skip_year_one=True)
    periodic_rates = calculate_discounting_rate_factor(discount_rate,start_year,end_year,maintain_period=period_freq,skip_year_one=True)

    return (sum(routine_rates)*routine_cost + sum(periodic_rates)*periodic_cost,
            list(routine_rates*routine_cost + periodic_rates*periodic_cost))

def adaptation_costs_npvs(benefit_costs,
                        asset_id_column,
                        flood_protection_name,
                        cost_type,
                        discount_rate=4.5,
                        start_year=2016,end_year=2080,
                        maintain_period=3):
    
    routine_rates = calculate_discounting_rate_factor(discount_rate,
                                                    start_year,end_year,
                                                    maintain_period=1,
                                                    skip_year_one=True)
    periodic_rates = calculate_discounting_rate_factor(discount_rate,
                                                    start_year,end_year,
                                                    maintain_period=benefit_costs[f'{cost_type}_periodic_freq'].values[0],
                                                    skip_year_one=True)

    benefit_costs[f'{cost_type}_total_maintenance_cost_npv'] = sum(routine_rates)*benefit_costs[
                                                                f'{cost_type}_routine_cost_{flood_protection_name}'
                                                                ] + sum(periodic_rates)*benefit_costs[
                                                                f'{cost_type}_periodic_cost_{flood_protection_name}']
    # benefit_costs[f'{cost_type}_total_maintenance_cost_npv'], \
    # benefit_costs[f'{cost_type}_total_maintenance_cost_npv_timeseries'] = zip(*benefit_costs.progress_apply(
    #                         lambda x: get_maintenance_costs(x[f'{cost_type}_routine_cost_{flood_protection_name}'],
    #                                                         x[f'{cost_type}_periodic_cost_{flood_protection_name}'],
    #                                                         discount_rate,start_year,end_year,
    #                                                         x[f'{cost_type}_periodic_freq']),
    #                                                         axis=1))                                    
    benefit_costs[f'{cost_type}_total_adapt_cost_npv'] = benefit_costs[f'{cost_type}_ini_adapt_cost_{flood_protection_name}'] + \
                                            benefit_costs[f'{cost_type}_total_maintenance_cost_npv']
    adapt_cost_columns = [f'{cost_type}_ini_adapt_cost_{flood_protection_name}',
                        f'{cost_type}_total_maintenance_cost_npv',
                        f'{cost_type}_total_adapt_cost_npv']

    return benefit_costs[[asset_id_column,flood_protection_name,
                        f'{cost_type}_ini_adapt_cost_{flood_protection_name}',
                        f'{cost_type}_total_maintenance_cost_npv',
                        f'{cost_type}_total_adapt_cost_npv']], adapt_cost_columns

def get_risk_and_adaption_columns(dataframe_columns):
    EAD_columns = [c for c in dataframe_columns if "EAD" in c]
    EAEL_columns = [c.replace("EAD","EAEL") for c in EAD_columns]
    benefit_columns = [c.replace("EAD","avoided_risk") for c in EAD_columns]
    bcr_columns = [c.replace("EAD","BCR") for c in EAD_columns]
    return EAD_columns, EAEL_columns, benefit_columns, bcr_columns


def get_adaptation_benefits(sector_protection_risks,sector_no_protection_risks,ignore_columns):
    benefit_columns = [c for c in sector_protection_risks.columns.values.tolist() if c not in ignore_columns]
    benefit_index_columns = [c for c in benefit_columns if "EAD" not in c and "EAEL" not in c]
    ead_columns = [c for c in benefit_columns if "EAD" in c]
    eael_columns = [c for c in benefit_columns if "EAEL" in c]
    ead_benefit_columns = [c.replace("EAD","Avoided_EAD") for c in ead_columns]
    eael_benefit_columns = [c.replace("EAEL","Avoided_EAEL") for c in eael_columns]
    sector_benefits = sector_protection_risks.copy()[benefit_columns].set_index(benefit_index_columns).fillna(0)
    sector_no_protection_risks = sector_no_protection_risks.set_index(benefit_index_columns).fillna(0)
    sector_benefits[ead_benefit_columns] = -1.0*(sector_benefits[ead_columns].sub(
                                            sector_no_protection_risks[ead_columns],
                                            axis='index',
                                            fill_value=0))
    sector_benefits[eael_benefit_columns] = -1.0*(sector_benefits[eael_columns].sub(
                                            sector_no_protection_risks[eael_columns],
                                            axis='index',
                                            fill_value=0))
    sector_benefits = sector_benefits.reset_index()
    # print (sector_benefits)
    sector_benefits = sector_benefits[benefit_index_columns + ead_benefit_columns + eael_benefit_columns]
    sector_protection_risks = pd.merge(sector_protection_risks,sector_benefits,how="left",on=benefit_index_columns)
    sector_no_protection_risks = sector_no_protection_risks.reset_index()

    return sector_protection_risks