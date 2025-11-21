from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Iterable, Callable

import numpy as np
import pandas as pd
import geopandas as gpd
import fiona

import osmnx as ox
from dotenv import load_dotenv

from ..universe import UniverseLoader
from ...utils.nyc_geocoding.normalizers import StreetNormalizer
from ...utils.pipeline_mixin import PipelineMixin
from ...utils.data_utils import flatten_array

load_dotenv()

logger = logging.getLogger('LIONLoader')

DATA_PATH = Path(str(os.getenv('DATA_PATH')))
LION_PATH =  DATA_PATH / 'raw/locations/lion/lion.gdb'

class LIONLoader(UniverseLoader, PipelineMixin):
    """Load universe locations from NYC LION dataset.

    Uses pipeline pattern for multi-step processing:
    1. Load base files (nodes, edges)
    2. Filter and clean data
    3. Convert to graph
    4. Consolidate intersections
    5. Assign street names
    6. Clean/normalize street names
    7. Convert to universe format
    """

    SOURCE = 'lion'
    CRS = 'EPSG:2263'

    def __init__(self, **kwargs: Any):
        """Initialize LIONLoader.

        Args:
            **kwargs: Optional parameters (currently unused, for compatibility with from_source())
        """
        pass

    def load_basefiles(self, lion_path:Path=LION_PATH) -> dict[str, gpd.GeoDataFrame]:
        # , layers:list[str]=fiona.listlayers(LION_PATH)
        return {
            'node': gpd.read_file(lion_path, layer='node', engine='pyogrio'),
            'node_stname': gpd.read_file(lion_path, layer='node_stname', engine='pyogrio'),
            'lion': gpd.read_file(lion_path, layer='lion', engine='pyogrio')
        }

    def filter_and_clean(self, base_gdfs: dict[str, gpd.GeoDataFrame]) -> dict[str, gpd.GeoDataFrame]:
        """
        Filter LION data to roads and active nodes

        Args:
            base_nodes: 'node' GeoDataFrame
            base_lion:  'lion' GeoDataFrame
        
        Returns:
            Dictionary containing filtered 'nodes' and 'edges' GeoDataFrames
        """ 
        base_nodes = base_gdfs['node']
        base_lion = base_gdfs['lion']

        # Filter to roads and walking paths only
        mask_roads = base_lion['FeatureTyp'].isin(['0']) # Mask only the roads
        mask_road_paths = base_lion['FeatureTyp'].isin(['0', 'W']) # Also mask the walking paths

        # Find only nodes connected to roads/paths
        active_nodes = pd.concat([
            base_lion.loc[mask_road_paths,'NodeIDFrom'],
            base_lion.loc[mask_road_paths, 'NodeIDTo']
        ]).unique().astype(int)
        
        filtered_nodes = base_nodes[base_nodes['NODEID'].isin(active_nodes)]

        # Filter edges to roads only
        filtered_edges = base_lion[mask_roads][[
            'Street','FeatureTyp','NodeIDTo', 'NodeIDFrom', 
            'SegmentTyp', 'TrafDir', 'geometry'
        ]].copy()
        filtered_edges.loc[:, 'NodeIDFrom'] = filtered_edges['NodeIDFrom'].astype(int)
        filtered_edges.loc[:, 'NodeIDTo'] = filtered_edges['NodeIDTo'].astype(int)

        filtered_gdfs = {'nodes': filtered_nodes, 'edges': filtered_edges}

        return filtered_gdfs

    def to_graph(self, filtered_data: dict[str, gpd.GeoDataFrame], save_path:Path|None=None, save_names:list[str]=['nodes.geojson','edges.geojson']) -> ox.MultiDiGraph:
        """Convert filtered nodes and edges to OSMnx graph

        Args:
            filtered_data: Dictionary with 'nodes' and 'edges' GeoDataFrames
            save_path (Path | None, optional): OPtional path to save indexed GeoDataFrames
            save_names (list[str], optional): _description_. Defaults to ['nodes.geojson','edges.geojson'].

        Returns:
            OSMnx MultiDiGraph
        """
        nodes = filtered_data['nodes']
        edges = filtered_data['edges']
        
        # Prepare edges with OSMnx-style multi-index (u, v, key)
        edges_indexed = (
            edges
            .explode(index_parts=True)
            .rename({'NodeIDFrom': 'u', 'NodeIDTo': 'v'}, axis=1)
            .set_index(['u','v'])
        )

        n = edges_indexed.groupby(level=[0,1]).cumcount()
        edges_indexed = edges_indexed.set_index(n.rename('key'), append=True)

        nodes_indexed = nodes.rename({'NODEID':'osmid'}, axis=1).set_index('osmid')
        
        # Prepare nodes with osmid index and x/y coordinates
        nodes_indexed['x'] = nodes_indexed['geometry'].x
        nodes_indexed['y'] = nodes_indexed['geometry'].y
        
        graph = ox.graph_from_gdfs(
            nodes_indexed,
            edges_indexed
        )

        if save_path:
            nodes_indexed.to_file(save_path / 'nodes.geojson')
            edges_indexed.to_file(save_path / 'edges.geojson')

        return graph

    def consolidate_intersections(self, graph: ox.MultiDiGraph, tolerance:int=50, simplify:bool=True, save_path:Path|None=None, save_names:list[str]=['nodes_consolidated.geojson','edges_consolidated.geojson']):
        """Consolidate nearby intersections into single nodes

        Args:
            graph (ox.MultiDiGraph): Input OSMnx graph
            tolerance (int, optional): Distance tolerance for consolidation
            simplify (bool, optional): Should the graph be simplified before consolidating?
            save_path (Path | None, optional): _description_. Defaults to None.
            save_names (list[str], optional): _description_. Defaults to ['nodes_consolidated.geojson','edges_consolidated.geojson'].
        """
        if simplify:
            graph = ox.simplify_graph(graph)
        
        graph_consolidated = ox.consolidate_intersections(graph, tolerance=tolerance, dead_ends=False)

        if save_path:
            nodes, edges = ox.graph_to_gdfs(graph_consolidated)
            nodes.to_file(save_path / save_names[0])
            edges.to_file(save_path / save_names[1])

        return graph_consolidated

    @staticmethod
    def _node_to_streetnames(edges:gpd.GeoDataFrame, street_col:str):        
        def _consolidate_streetnames(x, sort_by_freq:bool=True):
            flattened = flatten_array(np.array(x))
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

    def assign_streetnames(self, graph: ox.MultiDiGraph, street_col:str='Street', remove_deadends:bool=True):
        nodes, edges = ox.graph_to_gdfs(graph)

        node_to_streetnames = self._node_to_streetnames(edges=edges, street_col=street_col)
        nodes_w_streetnames = nodes.merge(
            node_to_streetnames, 
            left_index=True, right_index=True
        )

        # Remove those with less than 2
        if remove_deadends:
            nodes_w_streetnames = nodes_w_streetnames[nodes_w_streetnames[street_col].apply(len) > 1]
        
        return nodes_w_streetnames
    
    
    def clean_streetnames(self, nodes_w_streetnames: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        s_n = StreetNormalizer()
        nodes_cleaned = nodes_w_streetnames.copy()

        nodes_cleaned['Street'] = nodes_cleaned['Street'].apply(s_n.normalize_batch)

        return nodes_cleaned

    def convert_to_universe(self, nodes_cleaned: gpd.GeoDataFrame) -> gpd.GeoDataFrame: 
        def _get_column_list_index(x, i):
            if i < len(x):
                return x[i]
            else:
                return pd.NA
        nodes_cleaned_temp = nodes_cleaned.copy()
        nodes_cleaned_temp['street1'] = nodes_cleaned_temp['Street'].apply(_get_column_list_index, i=0)
        nodes_cleaned_temp['street2'] = nodes_cleaned_temp['Street'].apply(_get_column_list_index, i=1)

        # Rename and select columns
        nodes_converted = nodes_cleaned_temp.reset_index().rename(columns = {
            'index': 'location_id',
            'osmid_original': 'original_nodeids',
            'Street': 'additional_streets'
        })

        # Ensure original nodeids is a list
        nodes_converted['original_nodeids'] = nodes_converted['original_nodeids'].apply(
            lambda x: x if isinstance(x, list) else [x]
        )

        # Select Final Columns
        nodes_converted = nodes_converted[['location_id', 'street1','street2','additional_streets','original_nodeids','street_count','geometry']]

        # Convert to gdf and WGS84
        nodes_converted = gpd.GeoDataFrame(nodes_converted, crs=self.CRS).to_crs('EPSG:4326')

        return nodes_converted

    def _load_pipeline(
            self, 
            tolerance:int=30, 
            save_path:Path|None=None
        ) -> Iterable[tuple[str, Callable, list, dict[str, any]]]:
        """Define the processing pipeline for LION data.

        Each step receives the output of the previous step as its first argument

        Args:
            tolerance: Distance tolerance for consolidating intersections (meters)
            save_path: Optional path to save intermediate results

        Returns:
            List of pipeline steps: (name, function, kwargs)
        """
        pipeline = [
            ('Load Basefiles', self.load_basefiles, {}),
            ('Filter and Clean', self.filter_and_clean, {}),
            ('Convert to Graph', self.to_graph, {'save_path': save_path}),
            ('Consolidate Intersections', self.consolidate_intersections, {'tolerance': tolerance, 'save_path': save_path}),
            ('Assign Streetnames', self.assign_streetnames, {}),
            ('Clean Streetnames', self.clean_streetnames, {}),
            ('Convert to Universe', self.convert_to_universe, {})
        ]
        return pipeline

    def _load_raw(self, tolerance:int=30, save_path:Path|None=None) -> gpd.GeoDataFrame:
        """Load raw LION data by executing the pipeline.

        Args:
            tolerance: Distance tolerance for consolidating intersections (meters)
            save_path: Optional path to save intermediate results

        Returns:
            GeoDataFrame with universe location data in EPSG: 4326
        """
        return self._execute_pipeline(
            tolerance=tolerance,
            save_path=save_path
        )