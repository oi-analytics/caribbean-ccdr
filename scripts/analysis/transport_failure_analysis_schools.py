"""Plot traffic and rerouting for each school district.

Timings:
--------
Truncating paths with threshold 99%.
Number of paths before: 106,060.
Number of paths after: 14,008.
Took 31.2875 seconds.

"""
import os
os.environ['USE_PYGEOS'] = '0'
from os.path import join
import networkx as nx
from importlib import reload
import pandas as pd
import geopandas as gpd
from geospatial_utils import load_config
from num2words import num2words
import transport_flow_and_disruption_functions as tfdf


# global settings
COST = 'time_m'
THRESH = 60
TRUNC_THRESH = .99
ZETA = 1
EDGE_ATTRS = ['edge_id', 'length_m', 'time_m']
RECALCULATE_PATHS = True
RECALCULATE_TRAFFIC = False
COUNTRY = 'DMA'
plot_kwargs = {'dpi': 400, 'bbox_inches': 'tight'}
caribbean_epsg = 32620


def main(CONFIG):
    # set up directories
    indir, datadir, resultsdir, figdir = CONFIG['paths']['incoming_data'], CONFIG['paths']['data'], CONFIG['paths']['results'], CONFIG['paths']['figures']

    # load roads, roads network, and schools
    roads, road_net = tfdf.get_roads(join(datadir, 'infrastructure', 'transport'), COUNTRY, ['edge_id', 'length_m', 'time_m'])
    roads = roads.to_crs(epsg=caribbean_epsg)
    schools = gpd.read_file(join(datadir, 'infrastructure', 'social', 'education.gpkg'))
    schools = schools[schools['iso_code'] == COUNTRY]

    # load school districts (slightly different file formats)
    if COUNTRY == 'LCA':
        admin_areas = gpd.read_file(join(datadir, 'infrastructure', 'social', f'{COUNTRY.lower()}_edu_districts.gpkg'), layer='areas')[["DIS","geometry"]]
        admin_areas.loc[:, "DIS"] = admin_areas.apply(lambda x: str(num2words(x["DIS"])).title(), axis=1)
        admin_areas.rename(columns={"DIS": "school_district"},inplace=True)
    elif COUNTRY == "VCT":
        admin_areas = gpd.read_file(os.path.join(datadir, 'admin_boundaries', f'{COUNTRY.lower()}_edu_districts.gpkg'))[["SCHOOL_DIST","geometry"]]
        admin_areas["SCHOOL_DIST"] = admin_areas.apply(lambda x: str(num2words(x["SCHOOL_DIST"])).title(),axis=1)
        admin_areas.rename(columns={"SCHOOL_DIST": "school_district"},inplace=True)
    elif COUNTRY == "GRD":
        admin_areas = gpd.read_file(os.path.join(indir, 'admin_boundaries', f'gadm41_{COUNTRY}.gpkg'), layer='ADM_ADM_1')
        admin_areas.rename(columns={"NAME_1": "school_district"},inplace=True)
        admin_areas.loc[admin_areas['school_district'] == 'Carriacou', 'school_district'] = 'Carriacou & Petite Martinique'
    elif COUNTRY == "DMA":
        admin_areas = gpd.read_file(os.path.join(indir, 'admin_boundaries', f'gadm41_{COUNTRY}.gpkg'), layer='ADM_ADM_1')
        admin_areas.rename(columns={"NAME_1": "school_district"}, inplace=True)

    # TODO: GRD and DMA
    admin_areas = admin_areas.to_crs(epsg=caribbean_epsg)
    admin_areas.loc[:, "school_district"] = admin_areas["school_district"].astype(str).str.replace("Saint","St.")

    import pdb; pdb.set_trace()
    assert set(admin_areas['school_district'].unique()) == set(schools['school_district'].unique()), "School districts do not match"
    # step 1: get path and flux dataframe and save
    # path_df, school_road_net, roads_by_district = tfdf.process_school_fluxes(roads, road_net, schools, admin_areas, resultsdir, COUNTRY, COST, THRESH, ZETA, RECALCULATE_PATHS, TRUNC_THRESH)
    path_df, *_ = tfdf.process_school_fluxes_new(roads, road_net, schools, admin_areas, resultsdir, COUNTRY, COST, THRESH, ZETA, RECALCULATE_PATHS, TRUNC_THRESH)
    
    # path_df.describe()  # max time to get to any school: 40.79 mins, min: 0 mins

    # step 2: model disruption
    outfile = os.path.join(resultsdir, "transport", "disruption results", f"{COUNTRY.lower()}_schools_roads_edges_sector_damages_with_rerouting")
    disruption_file = os.path.join(resultsdir, "transport", "disruption results", f"{COUNTRY.lower()}_roads_edges_sector_damages.parquet")
    disruption_df = pd.read_parquet(disruption_file)
    disruption_df = tfdf.get_disruption_stats(disruption_df, path_df, road_net, COST)
    disruption_df.to_parquet(f"{outfile}.parquet")

    # step 3: add traffic to road edges
    # TODO: all code below here outdated
    if RECALCULATE_TRAFFIC:
        pathname = os.path.join(resultsdir, 'transport', 'traffic', f"{COUNTRY}_schools_traffic_{COST}_{THRESH}")
        traffic_df = tfdf.get_flow_on_edges(path_df, 'edge_id', 'edge_path', 'flux')
        traffic_df = traffic_df.rename(columns={'flux': 'traffic'})
        roads_traffic = roads.merge(traffic_df, how='left', on='edge_id')
        tfdf.test_traffic_assignment(roads_traffic, roads)  # check number of edges unchanged
        roads_traffic.to_file(filename=f"{pathname}.gpkg", driver="GPKG", layer="roads")

    # # plot example of school traffic
    # fig = tfdf.plot_district_traffic(roads_traffic, schools, 'Two', "school_district", "assigned_students")
    # fig.savefig(os.path.join(figdir, f"{COUNTRY}_schools_traffic_zoom_{COST}_{THRESH}.png"), **plot_kwargs)


    # # plot changes to traffic
    # fig = tfdf.plot_failed_edge_traffic(roads_disrupted, scenario_dict, edge_ix=2, buffer=1000)
    # fig.savefig(join(figdir, f"{COUNTRY}_schools_traffic_change_zoom_{scenario_dict['scenario_name']}_{COST}_{THRESH}.png"), **plot_kwargs)


if __name__ == "__main__":
    CONFIG = load_config(join("..", "..", ".."))
    main(CONFIG)