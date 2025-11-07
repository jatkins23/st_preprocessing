# functions for

from __future__ import annotations

from pathlib import Path
import os
import sys
from dataclasses import dataclass
from abc import ABC, abstractmethod

import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import fiona

from dotenv import load_dotenv
from ..universe import UniverseLoader

load_dotenv()

DATA_PATH = Path(str(os.getenv('DATA_PATH')))
LION_PATH =  DATA_PATH / 'raw/locations/lion/lion.gdb'
LION_LAYERS = {'nodes': 'nodes', 'node_names': 'node_stname', 'altnames': 'altnames','master': 'lion'}
TEMP = {'nodes': 'node','node_names': 'node_stname'}
layers_available = fiona.listlayers(LION_PATH)

# class StreetNames: 
#     pass
    
#     @classmethod
#     def clean_street_names(cls):

class LocationsLIONLoader(UniverseLoader):
    SOURCE = 'lion'

    def __init__(self, path: str, *, encoding: str|None=None) -> None:
        self.path = path
        self.encoding = encoding

    def _load_baselayers(lion_path:Path=LION_PATH, layers: dict[str, str]=LION_LAYERS) -> dict[str, gpd.GeoDataFrame]:
        layers_dict = {}
        for lyr_name, lyr_path in layers.items():
            try:
                layers_dict[lyr_name] = gpd.read_file(lion_path, layer=lyr_path)
            except Exception as e:
                raise ValueError(f'{lyr_path} not found! Only {", ".join(list(layers_available))}')
            
        return layers_dict
            
    def _clean_streetnames(x): # TODO: improve to actualy work correctly
        if not isinstance(x, str):
            return pd.NA
        elif (x.endswith(' BOUNDARY')) or (' RAIL' in x) or ('SHORELINE' in x):
            return pd.NA
        else:
            return x.strip()
        
    # 1) Load the raw lion files
    # 2) Handle StreetNames:
    #   2.1) Filter streets based on type.
    #   2.2) Clean the street names
    #   2.3) Remove null street names
    # 3) Merge the nodes and street names
        # This should filter out the unnecessary nodes
        #     
        
    def _load_raw(self):
        # load in raw layers
        layers = self._load_baselayers(LION_PATH, {'nodes': 'node','node_names': 'node_stname'})

        # Handle StreetNames
        
        print('loading from LION')
        return pd.DataFrame({
            # 'A': [1,2,3],
            # 'B': [4,5,6],
            'name': ['nyc','nyc','nyc'],
            'source': ['lion','lion','lion'],
        })
