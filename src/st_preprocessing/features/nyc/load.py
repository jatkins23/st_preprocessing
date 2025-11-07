from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely import from_wkt
from pyproj import Proj

def load_standard(path:Path, crs:Proj|str|int='EPSG:4326') -> gpd.GeoDataFrame:
    temp = pd.read_csv(path)
    ret_gdf = gpd.GeoDataFrame(temp, geometry=from_wkt(temp['the_geom']), crs=crs)
    
    return ret_gdf

    # gdfs['raised_xwalk'] = gpd.GeoDataFrame(dfs['raised_xwalk'], geometry=from_wkt(dfs['raised_xwalk']['WKT Geometry']))
    # dfs['raised_xwalk'] = None

    # bollards
    # dfs['bollards'] # only address. Could use overpass or something but not worth it

    # bus_pad 
    # dfs['bus_pad'] # same as above. Has cross stretes but probably not worth it for 325 or so entries rn.

    # cap_projx
    # dfs['cap_projx'] # No location. It has a name though which might be helpful idk. But skipping

    # speed_humps
    #dfs['speed_humps'] # About 49 are actually malformed

    #gdfs['speed_humps'] = gpd.GeoDataFrame(dfs['speed_humps'], geometry = dfs['speed_humps']['the_geom'].apply(safe_load_wkt))
    
