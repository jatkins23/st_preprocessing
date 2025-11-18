# Source data for loading and cleaning capital reconstruction projects from NYC OpenData
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import pandas as pd
import geopandas as gpd
from pyproj import Proj
from shapely import from_wkt

from ..project_loader import ProjectLoader

DATA_PATH = Path(str(os.getenv('DATA_PATH')))

OPENNYC_DATA_PATH = DATA_PATH / 'raw' / 'citydata' / 'openNYC'
CORE_FILE_NAME = Path('Street_and_Highway_Capital_Reconstruction_Projects_-_Intersection_20250721.csv')

COLUMNS_TO_KEEP = ['ProjectID','ProjTitle', 'FMSID', 'FMSAgencyID',
       'LeadAgency', 'Managing Agency', 'ProjectDescription',
       'ProjectTypeCode', 'ProjectType', 'ProjectStatus', 'ConstructionFY',
       'DesignStartDate', 'ConstructionEndDate', 'CurrentFunding',
       'ProjectCost', 'OversallScope', 'SafetyScope', 'OtherScope',
       'ProjectJustification', 'OnStreetName', 'FromStreetName',
       'ToStreetName', 'OFTCode', 'DesignFY','geometry']
DROP_SCOPES = ['Wayfindiing', 'Resurfacing', 'Median-Planted Trees', 'DEP Project', 'Percent for Art', 'GI in Scope', 'Potential GI']


def load_standard(path:Path, crs:Proj|str|int='EPSG:4326') -> gpd.GeoDataFrame:
    temp = pd.read_csv(path)
    ret_gdf = gpd.GeoDataFrame(temp, geometry=from_wkt(temp['the_geom']), crs=crs)
    
    return ret_gdf

class ProjectLoaderNYC(ProjectLoader):
    def load_caprecon_file(universe:str='nyc', data_path:Path=OPENNYC_DATA_PATH, source_file_name:Path=CORE_FILE_NAME, input_crs:Proj='EPSG:4326') -> gpd.GeoDataFrame:
       source_file_path = data_path / source_file_name

       projects_gdf = pd.read_csv(source_file_path)
       projects_gdf = gpd.GeoDataFrame(projects_gdf, geometry=from_wkt(projects_gdf['the_geom']), crs=input_crs)
       filtered_caprecon_gdf = projects_gdf[projects_gdf['ProjectType'] == 'CAPITAL RECONSTRUCTION'][COLUMNS_TO_KEEP]

       return filtered_caprecon_gdf
       
#     def clean_file(caprecon_gdf):
#        caprecon_gdf = load_caprecon_file('nyc')

#        pd.to_datetime(caprecon_gdf['ConstructionEndDate'], errors='coerce').dropna() # 2799 -> 1482
#        caprecon_gdf[caprecon_gdf['ConstructionEndDate'] == '0000/00/00'] # 1317 have all 00s # But they have construction fiscal_year

#        # So we go with:
#        construction_date = pd.to_datetime(caprecon_gdf['ConstructionEndDate'], errors='coerce')
#        pd.to_datetime(caprecon_gdf['ConstructionFY'])

#        constrcution_fy_datetime = pd.to_datetime(caprecon_gdf[caprecon_gdf['ConstructionFY'] != 0]['ConstructionFY'], format='%Y')

#        project_datetime = construction_date.fillna(constrcution_fy_datetime)
#        caprecon_gdf['project_datetime'] = project_datetime

#        def filter_scopes(caprecon_gdf:gpd.GeoDataFrame):
#        #caprecon_gdf['OversallScope'].str.split(',').explode().value_counts(dropna=False) 
#        # Overall Scope is just 4 types: Sidewalks, Curb to Curb Reconstruction, Partial Reconstruction, Resurfacing
#        # caprecon_gdf[caprecon_gdf['OversallScope'].isna()][['OtherScope','SafetyScope']]['SafetyScope'].str.split(',').explode().value_counts()


#        # Include: All in OversallScope
#        all_overall = list(caprecon_gdf['OversallScope'].str.split(',').explode().value_counts().index.values)
#        allowed_overall = ['Sidewalks', 'Curb to Curb Reconstruction', 'Partial Reconstruction']

#        # Include: All in Safety Scope
#        all_safety = list(caprecon_gdf['SafetyScope'].str.split(',').explode().value_counts().index.values)
#        caprecon_gdf['OtherScope'].str.split(',').explode().value_counts() 
#        # In Other Scope: only Plaza/Ped Space Enhancement

#        ALLOWED_FEATURES = all_overall + all_safety + ['Plaza/Ped Space Enhancement']

#        all_scopes = pd.concat([caprecon_gdf['OversallScope'], caprecon_gdf['SafetyScope'], caprecon_gdf['OtherScope']]).str.split(',').explode()
#        allowed_scopes = all_scopes[all_scopes.isin(ALLOWED_FEATURES)]
#        caprecon_gdf['total_scope'] = allowed_scopes.groupby(level=0).agg(list)
#        caprecon_gdf['proj_year'] = caprecon_gdf['project_datetime'].dt.year
#        caprecon_gdf[['ProjectID', 'ProjTitle', 'LeadAgency', 'ProjectType', 'P

def load_nyc_file():
    pass    

