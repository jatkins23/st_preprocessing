from __future__ import annotations

import os
import logging
from pathlib import Path

import pandas as pd
import geopandas as gpd
import fiona

import osmnx as ox
from dotenv import load_dotenv

from ..universe import UniverseLoader

load_dotenv()

DATA_PATH = Path(str(os.getenv('DATA_PATH')))
LION_PATH =  DATA_PATH / 'raw/locations/lion/lion.gdb'

class LionLoader(UniverseLoader):
    def __init__(self, verbose:bool=False):
        self.verbose=verbose
        super().__init__()

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

    def to_graph(self, nodes:gpd.GeoDataFrame, edges:gpd.GeoDataFrame, save_path:Path|None=None, save_names:list[str]=['nodes.geojson','edges.geojson']) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
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

    def assign_streetnames(self, nodes, edges, street_col:str='Street'):
        # Node Streetnames
        def _node_to_streetnames(edges:gpd.GeoDataFrame, street_col):
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

        _node_to_streetnames = _node_to_streetnames(edges=edges)
        nodes_w_streetnames = nodes.merge(node_streetnames, left_index=True, right_index=True)
        nodes_w_streetnames[nodes_w_streetnames[street_col].apply(len) > 1]
        
        return nodes_w_streetnames

    def _load_raw(self, tolerance:int=30, save_path:Path|None=None):
        pipeline = [
            ('Load Basefiles', self.load_basefiles, [], {}),
            ('Filter and Clean', self.filter_and_clean, [], {}),
            ('Convert to Graph', self.filter_and_clean, [self.filtered_nodes, self.filtered_edges], {'save_path': save_path}),
            ('Consolidate Intersections', self.consolidate_intersections, [], {'tolerance': tolerance, 'save_path': save_path}),
            ('Assign Streetnames', assign_streetnames, [], {})    
        ]
        #results = {}
        for name, func, args, kwargs in pipeline:
            try:
                func(*args, **kwargs)
                # results[name] = result
                if verbose:
                    print(f'{name}: Completed.')
            except Exception as e:
                print(f'{name} failed: {e}')

if __name__ == '__main__':
    TEMP_FILES = Path('/Users/jon/code/st_preprocessing/data/temp')
    ll = LionLoader()
    ll._load_raw(save_path=TEMP_FILES, verbose=True)
    
    
    #nodes_consolidated, edges_consolidated = ox.graph_to_gdfs(ll.graph_consolidated).to_parquet(TEMP_FILES / 'consolidated.')