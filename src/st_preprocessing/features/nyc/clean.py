import pandas as pd
import geopandas as gpd

def clean_bike_rtes(input_gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return_gdf = input_gdf.drop(columns=['the_geom','segmentid','version','bikeid','prevbikeid','boro','street','fromstreet','tostreet','ft2facilit','tf2facilit', 'Shape_Leng'])
    return_gdf['install_date'] = pd.to_datetime(return_gdf['instdate'])
    return_gdf['removal_date'] = pd.to_datetime(return_gdf['ret_date'])
    return_gdf = return_gdf.drop(columns=['instdate', 'ret_date'])
    return_gdf[return_gdf['facilitycl'] != return_gdf['allclasses']]

    return_gdf[['lanecount', 'bikedir']].value_counts(dropna=False)
    return_gdf = return_gdf.drop(columns=['allclasses'])
    return_gdf = return_gdf.rename(columns = {'facilitycl':'class', 'lanecount':'lanes', 'install_date': 'installdate','removal_date':'removedate', 'ft_facilit':'facility'})
    return_gdf['offstreet'] = return_gdf['onoffst'] == 'OFF'
    return_gdf = return_gdf[['status','offstreet','class','lanes','facility','bikedir','lanes','installdate','removedate','geometry']]

    return return_gdf

def clean_bus_lanes(input_gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    DROP_COLS = ['Street','the_geom','StreetWidt','Boro','Facility','Hours','Days','RW_TYPE','TrafDir','SegmentID','Days_Code','Shape_Leng','Shape_Le_1','Chron_ID_1','SBS_Route1','SBS_Route2','SBS_Route3']
    return_gdf = input_gdf.drop(columns=DROP_COLS)
    return_gdf[return_gdf['Year3'].notna()]
    return_gdf['lastupdate'] = pd.to_datetime(return_gdf['Last_Updat'], errors='coerce').combine_first(pd.to_datetime(return_gdf['Last_Updat'], errors='coerce', format='%m/%d/%y'))
    #return_gdf[return_gdf['lastupdate'].isna()][['lastupdate','Last_Updat']]
    # return_gdf[return_gdf['Open_dates'] != return_gdf['Last_Updat']] # THey are all the same
    return_gdf[['Lane_Type1','Lane_Type2']].value_counts(dropna=False).reset_index()
    # Drop: ['Lane_Type2','Late_Updat']
    return_gdf = return_gdf.drop(columns=['Lane_Type2','Last_Updat','Open_dates'])
    return_gdf = return_gdf.rename(columns={'Lane_width': 'width', 'Lane_Type1':'type','Lane_Color': 'color'}).rename(columns=lambda x: x.lower())

    return return_gdf

def clean_ped_plaza(input_gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    KEEP_COLS = ['PlazaName','OnStreet','geometry']
    return_gdf = input_gdf[KEEP_COLS].rename({'PlazaName': 'name', 'OnStreet':'onstreet'})
    
    return return_gdf

def clean_traffic_calming(input_gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # `traffic_calming`
    return_gdf = input_gdf[['treatment_','geometry', 'completion']].copy()
    return_gdf['completion'] = pd.to_datetime(return_gdf['completion'], format='%m/%d/%Y %I:%M:%S %p')
    return_gdf = return_gdf.rename(columns={'treatment_':'treatment','completion':'install_date'})
    
    return return_gdf

