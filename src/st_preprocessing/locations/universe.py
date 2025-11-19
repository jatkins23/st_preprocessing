# Abstract Class for a LocationSet, which contains necessary 
# -- Maybe rename to 'Region'
from __future__ import annotations

from typing import Any, ClassVar, Type, Iterable, Mapping
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from pydantic import BaseModel, ValidationError

from ..utils.errors import DataValidationError
from .location import Location

logger = logging.getLogger('UniverseLoader')

class Universe(BaseModel):
    name: str # nyc
    source: str # nyc
    locations: Iterable[Location] # gdf
    images: Path # to duckdb table


class UniverseLoader(ABC):
    """Abstract base for Universe loaders.
    Subclasses will set a SOURCE and implement a `_load_raw()` function to return
    the necessary data objects
    """

    # Unique key for each subclass
    SOURCE = ClassVar[str]

    # Global registry of loaders
    _REGISTRY: ClassVar[dict[str, Type['UniverseLoader']]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        
        # Auto-register subclasses that define a SOURCE string
        source = getattr(cls, "SOURCE", None)
        if source:
            key = str(source).lower()
            if key in UniverseLoader._REGISTRY and UniverseLoader._REGISTRY[key] is not cls:
                raise RuntimeError(f"Duplicate loader SOURCE '{key}' for {cls.__name__}")
            UniverseLoader._REGISTRY[key] = cls

    @classmethod
    def from_source(cls, source: str, **kwargs: Any):# -> pd.DataFrame:
        """Factor to dispact registered subclass by `source`."""

        key = str(source).lower()
        try: 
            loader_cls = cls._REGISTRY[key]
        except KeyError as e:
            raise ValueError(
                f"Unknown source '{source}'."
                f"Known: {sorted(cls._REGISTRY.keys())}"
            ) from e
        
        loader = loader_cls(**kwargs)
        return loader.load()

    def _validate(self, rows:Iterable) -> None:
        try: 
            models: list[Location] = [Location.model_validate(r, strict=False, by_name=True) for r in rows]
        except ValidationError as e:
            logger.error(
                'Universe validation failed',
                extra={"source": self.SOURCE, "error_count": len(e.errors())}
            )
            raise DataValidationError(self.SOURCE, e.errors(), original=e)
        
        return models

    def load(self) -> pd.DataFrame:
        """
        High-level load: subclasses provide raw records;
        base class validates & normalizes to a consistent Dataframe
        Right now it just prints
        """
        # Load Raw
        raw = self._load_raw()
        
        # Validate and coerce
        rows = raw.to_dict(orient="records") if isinstance(raw, pd.DataFrame) else list(raw)
        model_instances = self.validate(rows)

        # Convert models to plain dicts and build a normalized DataFrame
        return pd.DataFrame([m.model_dump() for m in model_instances])

    @staticmethod
    def interpret_boundary(universe_boundary:str|Path|Polygon|gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if isinstance(universe_boundary, (Polygon, gpd.GeoDataFrame)):
            bounds=universe_boundary
        elif isinstance(universe_boundary, str) and os.path.exists(universe_boundary):
            bounds = gpd.read_file(Path(str(universe_boundary)))
        elif str(universe_boundary) in ['nyc','all']:
            bounds = None
        else: # Geocodable address
            bounds = ox.geocoder.geocode_to_gdf(str(universe_boundary))

        return bounds

    def clip_by_boundary(self, universe_boundary):
        bounds = self.interpret_boundary(universe_boundary)
        if bounds is not None:
            locations_clipped = self.locations.clip(
                bounds.to_crs(self.locations.crs)
            )
        else:
            locations_clipped = self.locations

        return locations_clipped

    @abstractmethod
    def _load_raw(self) -> Iterable[Location | Mapping[str, Any]]:
        ...
        



class NYC_Universe(Universe):
    modalities = 