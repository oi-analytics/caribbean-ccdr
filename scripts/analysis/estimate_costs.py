import pandas as pd
import matplotlib.pyplot as plt

#i use shapely and geopandas only to show the coastline
import geopandas as gpd
import shapely
import numpy as np
import os

rcp = "ssp585"
rp= 50
rp_landslide= 1
epoch= 2030
rcp_landslide = 'baseline'
sh_name= "sdg"

country = ['dma', 'vct','grd','lca']
country = ['dma']

base = "/Users/raghavpant/Desktop/OIA_projects/Carribbean_CCDR/results/adaptation_outcomes/"
folder_stub ="_sector_specific_results"
file_stubs = ["_energy_risks_investments","_education_risks_investments","_roads_risks_investments","_ports_risks_investments","_airports_risks_investments","_health_risks_investments","_wtp_risks_investments","_wwtp_risks_investments"]
#"_roads_risks_investments","_energy_risks_investments","_ports_risks_investments","_airports_risks_investments","_health_risks_investments",
#sum_of_adaptation_investment_mean
#sum_of_new_build_investment_mean
#existing_asset_adaptation_implemented
#service_resilience_achieved_percentage

# resilience_threshold = [10,20,30,40,50,60,70,80,90,95,100]
resilience_threshold = [50,75,90,100]

for c in country:
    dfs=[]
    for f in file_stubs:
        for r in resilience_threshold:  
            #change directory
            # dir = base+"{co}{f_stub}\\{co}{f_stub}".format(co=c,f_stub=folder_stub)
            dir = os.path.join(base,f"{c}{folder_stub}")
            os.chdir(dir)

            #for each country, sector and resilience threshold, read the files in, subset to the min value that achieves a specific level of service resilience,
            #and then sum new infrastructure and resilience investments (where they exist). append sectoral data together 
            try:
                data = pd.read_excel('{co}{fs}.xlsx'.format(co=c, fs=f), sheet_name=sh_name)
                data = data[data.service_resilience_achieved_percentage>=r]

                data = data[((data["rcp"]==rcp) | (data["rcp"]==rcp_landslide)) & ((data["rp"]==rp) | (data["rp"]==1)) & (data["epoch"]==epoch)]
                # group by hazard, sector, and investment, and find the minimum value of service_resilience_percentage for each group
                try:
                    data['total_investment_mean'] = data['adaptation_investment_mean'] + data['new_build_resilience_investment_mean']
                    data = data[['subsector','hazard','rp','rcp','epoch',
                                'existing_asset_adaptation_implemented','total_investment_mean', 'service_resilience_achieved_percentage','damage_mean']]
                    
                    min = data[data['service_resilience_achieved_percentage'] >= r].groupby(['hazard', 'subsector', 'rp', 'rcp']).agg({
                    'service_resilience_achieved_percentage': 'min'}).reset_index()
                    data = min.merge(data, on=['hazard', 'subsector', 'rp', 'rcp','service_resilience_achieved_percentage'], how='left')
                    data ["resilience"] = r         
                    # merged_df = pd.merge(min_resilience, data, on=['hazard', 'subsector', 'adaptation_investment_mean', 'service_resilience_achieved_percentage'], how='left')

                except:
                    data['total_investment_mean'] = data['adaptation_investment_mean'] 
                    data = data[['subsector','hazard','rp','rcp','epoch',
                        'existing_asset_adaptation_implemented','total_investment_mean','service_resilience_achieved_percentage','damage_mean']]


                    min = data[data['service_resilience_achieved_percentage'] >= r].groupby(['hazard', 'subsector', 'rp', 'rcp']).agg({
                    'service_resilience_achieved_percentage': 'min'}).reset_index()
                    data = min.merge(data, on=['hazard', 'subsector', 'rp', 'rcp','service_resilience_achieved_percentage'], how='left')
                    data ["resilience"] = r
                dfs.append(data) 

            except:
                print(f+" does not exist") 

    dfs = pd.concat(dfs,axis=0,ignore_index=True)
    dfs.to_csv(base+ "{co}{f_stub}".format(co=c,f_stub=folder_stub)+'\\resilience_scenarios_2050.csv')

    #aggregate over sector
    dfs= dfs.reset_index()
    dfs_sectors = dfs.groupby(['hazard', 'resilience']).agg({'total_investment_mean':'sum','service_resilience_achieved_percentage':'mean','damage_mean':'sum'}).reset_index()
    dfs_sectors.to_csv(base+ "{co}{f_stub}".format(co=c,f_stub=folder_stub)+'\\resilience_scenarios_rp50_2030_sectoral.csv')

    #Aggregate over hazard
    dfs_hazard = dfs.groupby(['subsector', 'rp','resilience']).agg({'total_investment_mean':'sum','service_resilience_achieved_percentage':'mean','damage_mean':'sum'}).reset_index()
    dfs_hazard.to_csv(base+ "{co}{f_stub}".format(co=c,f_stub=folder_stub)+'\\resilience_scenarios_rp50_2030_hazard.csv')

    #Aggregate over hazard no landslide
    dfs_hazard = dfs.groupby(['subsector', 'rp','resilience']).agg({'total_investment_mean':'sum','service_resilience_achieved_percentage':'mean','damage_mean':'sum'}).reset_index()
    dfs_hazard.to_csv(base+ "{co}{f_stub}".format(co=c,f_stub=folder_stub)+'\\resilience_scenarios_rp50_2030_hazard_no_landslide.csv')

    #full aggregation across hazard and sector
    #Aggregate by hazard
    dfs_agg = dfs.groupby(['resilience']).agg({'total_investment_mean':'sum','service_resilience_achieved_percentage':'mean','damage_mean':'sum'}).reset_index()
    dfs_agg.to_csv(base+ "{co}{f_stub}".format(co=c,f_stub=folder_stub)+'\\resilience_scenarios_rp50_2030_aggregated.csv')

    #agggregate w/o landslide
    dfs_agg = dfs.groupby(['rp','resilience']).agg({'total_investment_mean':'sum','service_resilience_achieved_percentage':'mean','damage_mean':'sum'}).reset_index()
    dfs_agg.to_csv(base+ "{co}{f_stub}".format(co=c,f_stub=folder_stub)+'\\resilience_scenarios_rp50_2030_aggregated_no_landslide.csv')


    print(c+" done")

