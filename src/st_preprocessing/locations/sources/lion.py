from __future__ import annotations

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import fiona

import osmnx as ox
from dotenv import load_dotenv

from colorama import Fore, Style

from ..universe import UniverseLoader

load_dotenv()

DATA_PATH = Path(str(os.getenv('DATA_PATH')))
LION_PATH =  DATA_PATH / 'raw/locations/lion/lion.gdb'

class LIONLoader(UniverseLoader):
    SOURCE = 'lion'
    
    def __init__(self, verbose:bool=False):
        self.verbose=verbose

    def load_basefiles(self, lion_path:Path=LION_PATH) -> dict[str, gpd.GeoDataFrame]:
        # , layers:list[str]=fiona.listlayers(LION_PATH)
        self.node = gpd.read_file(lion_path, layer='node', engine='pyogrio')
        self.node_stname = gpd.read_file(lion_path, layer='node_stname', engine='pyogrio')
        self.lion = gpd.read_file(lion_path, layer='lion', engine='pyogrio')

    def filter_and_clean(self): 
        # Masks to filter LION
        mask_roads = self.lion['FeatureTyp'].isin(['0']) # Mask only the roads
        mask_road_paths = self.lion['FeatureTyp'].isin(['0', 'W']) # Also mask the walking paths

        active_nodes = pd.concat([self.lion.loc[mask_road_paths,'NodeIDFrom'], self.lion.loc[mask_road_paths, 'NodeIDTo']]).unique().astype(int)
        filtered_nodes = self.node[self.node['NODEID'].isin(active_nodes)]

        filtered_edges = self.lion[mask_roads][['Street','FeatureTyp','NodeIDTo', 'NodeIDFrom', 'SegmentTyp', 'TrafDir', 'geometry']]
        filtered_edges.loc[:, 'NodeIDFrom'] = filtered_edges['NodeIDFrom'].astype(int)
        filtered_edges.loc[:, 'NodeIDTo'] = filtered_edges['NodeIDTo'].astype(int)

        self.filtered_nodes = filtered_nodes
        self.filtered_edges = filtered_edges

    def to_graph(self, nodes:gpd.GeoDataFrame=None, edges:gpd.GeoDataFrame=None, save_path:Path|None=None, save_names:list[str]=['nodes.geojson','edges.geojson']) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        if not nodes:
            nodes = self.filtered_nodes
        if not edges:
            edges = self.filtered_edges
        edges_indexed = (
            edges
            .explode(index_parts=True)
            .rename({'NodeIDFrom': 'u', 'NodeIDTo': 'v'}, axis=1)
            .set_index(['u','v'])
        )

        n = edges_indexed.groupby(level=[0,1]).cumcount()
        edges_indexed = edges_indexed.set_index(n.rename('key'), append=True)

        nodes_indexed = nodes.rename({'NODEID':'osmid'}, axis=1).set_index('osmid')
        nodes_indexed['x'] = nodes_indexed['geometry'].x
        nodes_indexed['y'] = nodes_indexed['geometry'].y

        # self.nodes_indexed = nodes_indexed
        # self.edges_indexed = edges_indexed
        
        graph = ox.graph_from_gdfs(
            nodes_indexed,
            edges_indexed
        )

        if save_path:
            nodes_indexed.to_file(save_path / save_names[0])
            edges_indexed.to_file(save_path / save_names[1])

        self.graph = graph
        return graph

    def consolidate_intersections(self, tolerance:int=50, simplify:bool=True, save_path:Path|None=None, save_names:list[str]=['nodes_consolidated.geojson','edges_consolidated.geojson']):
        if simplify:
            g = ox.simplify_graph(self.graph)
        else:
            g = self.graph
        
        self.graph_consolidated = ox.consolidate_intersections(self.graph, tolerance=tolerance, dead_ends=False)    
        if save_path:
            nodes, edges = ox.graph_to_gdfs(self.graph_consolidated)
            nodes.to_file(save_path / save_names[0])
            edges.to_file(save_path / save_names[1])

    def assign_streetnames(self, nodes=None, edges=None, street_col:str='Street'):
        if not nodes:
            nodes, _ = ox.graph_to_gdfs(self.graph_consolidated)
        if not edges:
            _, edges = ox.graph_to_gdfs(self.graph_consolidated)

        # Node Streetnames
        def _node_to_streetnames(edges:gpd.GeoDataFrame, street_col:str):
            def _flatten_array(x):
                """
                Flattens a numpy array that may contain nested lists or arrays.
                Works recursively and returns a 1D numpy array.
                """
                # Convert to a Python list first (handles both np.ndarray and list)
                if isinstance(x, np.ndarray):
                    x = x.tolist()
                
                # Recursively flatten any nested lists
                def _flatten(lst):
                    for i in lst:
                        if isinstance(i, (list, np.ndarray)):
                            yield from _flatten(i)
                        else:
                            yield i
            
                return np.array(list(_flatten(x)), dtype=object)
            
            def _consolidate_streetnames(x, sort_by_freq:bool=True):
                flattened = _flatten_array(np.array(x))
                if sort_by_freq:
                    ret = np.array(pd.Series(flattened).value_counts().index.to_list())
                else:
                    ret = np.unique(flattened)
                return ret
            
            nodes_streetnames = pd.concat([
                edges.reset_index()[['u', street_col]].rename({'u':'Node'}, axis=1),
                edges.reset_index()[['v', street_col]].rename({'v':'Node'}, axis=1)
            ]).groupby('Node').agg({street_col: _consolidate_streetnames})
        
            return nodes_streetnames

        node_to_streetnames = _node_to_streetnames(edges=edges, street_col=street_col)
        nodes_w_streetnames = nodes.merge(node_to_streetnames, left_index=True, right_index=True)
        nodes_w_streetnames[nodes_w_streetnames[street_col].apply(len) > 1]

        self.nodes_w_streetnames = nodes_w_streetnames
        
        return nodes_w_streetnames
    
    def convert_to_universe(self): 
        def _get_column_list_index(x, i):
            if i < len(x):
                return x[i]
            else:
                return pd.NA
        temp = self.nodes_w_streetnames.copy()
        temp['street1'] = temp['Street'].apply(_get_column_list_index, i=0)
        temp['street2'] = temp['Street'].apply(_get_column_list_index, i=1)

        nodes_converted = temp.reset_index().rename(columns = {
            'index': 'location_id',
            'osmid_original': 'original_nodeids',
            'Street': 'additional_streets'
        })

        nodes_converted['original_nodeids'] = nodes_converted['original_nodeids'].apply(lambda x: x if isinstance(x, list) else [x])

        nodes_converted = nodes_converted[['location_id', 'street1','street2','additional_streets','original_nodeids','street_count','geometry']]
        # nodes_converted = nodes_converted[['location_id','street1','street2']]#,'geometry']]

        return nodes_converted

    def _load_raw(self, tolerance:int=30, save_path:Path|None=None, verbose:bool=True):
        pipeline = [
            ('Load Basefiles', self.load_basefiles, [], {}),
            ('Filter and Clean', self.filter_and_clean, [], {}),
            ('Convert to Graph', self.to_graph, [], {'save_path': save_path}),
            ('Consolidate Intersections', self.consolidate_intersections, [], {'tolerance': tolerance, 'save_path': save_path}),
            ('Assign Streetnames', self.assign_streetnames, [], {}),
            ('Convert to Universe', self.convert_to_universe, [], {})
        ]
        results = {}
        for name, func, args, kwargs in pipeline:
            try:
                results[name] = func(*args, **kwargs)
                if verbose:
                    arr_len = len('Consolidate Intersections') - len(name) + 4
                    print(f'LION -- {name} {'-' *  arr_len}> {Fore.GREEN}Completed{Style.RESET_ALL}.')
            except Exception as e:
                print(f'LION -- {name} {'-' *  arr_len}> {Fore.RED}Failed{Style.RESET_ALL}: {e}')
        
        return results[[x[0] for x in pipeline][-1]]


# @dataclass
# class LionState:
#     node: Optional[gpd.GeoDataFrame] = None
#     node_stname: Optional[gpd.GeoDataFrame] = None
#     lion: Optional[gpd.GeoDataFrame] = None
#     graph: Optional[Graph] = None
#     intersections: Optional[gpd.GeoDataFrame] = None
#     universe: Optional[Any] = None