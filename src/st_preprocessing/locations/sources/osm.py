from dataclasses import dataclass
from ..universe import UniverseLoader
import pandas as pd

class OSMLoader(UniverseLoader):
    SOURCE = 'osm'

    def __init__(self, path: str, *, encoding: str|None=None) -> None:
        self.path = path
        self.encoding = encoding

    def _load_raw(self):
        print('loading from OSM')
        return pd.DataFrame({
            'name': ['World','World','World'],
            'source': ['osm','osm','osm'],
        })
