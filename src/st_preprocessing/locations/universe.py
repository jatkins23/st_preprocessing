# Abstract Class for a LocationSet, which contains necessary 
# -- Maybe rename to 'Region'
from __future__ import annotations

from typing import Any, ClassVar, Type, Iterable, Mapping, Callable
import logging
import os
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from pydantic import BaseModel, ValidationError

from ..utils.errors import DataValidationError
from .location import Location
from ..data_loader import DataLoader

logger = logging.getLogger('UniverseLoader')

class Universe(BaseModel):
    name: str # nyc
    source: str # nyc
    locations: Iterable[Location] # gdf
    images: Path # to duckdb table


class UniverseLoader(DataLoader):
    """Abstract base for Universe loaders.

    Registers as 'universe' modality in DataLoader.
    Subclasses will set a SOURCE and implement a `_load_raw()` function to return
    the necessary data objects.

    Usage:
        # Direct usage
        data = UniverseLoader.from_source('nyc')

        # Via DataLoader
        data = DataLoader.load(modality='universe', source='nyc')
    """

    # Register this loader as the 'universe' modality
    MODALITY: ClassVar[str] = "universe"

    # Unique key for each subclass (e.g., 'nyc', 'sf', 'boston')
    SOURCE: ClassVar[str]

    # Global registry of source loaders
    _REGISTRY: ClassVar[dict[str, Type['UniverseLoader']]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Auto-register subclasses that DIRECTLY DEFINE a SOURCE string
        # (not inherited from parent class)
        if "SOURCE" in cls.__dict__:  # Only if defined on this class, not inherited
            source = cls.SOURCE
            key = str(source).lower()
            if key in UniverseLoader._REGISTRY and UniverseLoader._REGISTRY[key] is not cls:
                raise RuntimeError(f"Duplicate loader SOURCE '{key}' for {cls.__name__}")
            UniverseLoader._REGISTRY[key] = cls
            logger.debug(f"Registered UniverseLoader: {cls.__name__} as '{key}'")

    def load(self) -> pd.DataFrame:
        """High-level load: subclasses provide raw records;
        base class validates & normalizes to a consistent DataFrame.

        Returns:
            Validated and normalized DataFrame of locations
        """
        # Load raw data
        raw = self._load_raw()

        # Validate and coerce
        rows = raw.to_dict(orient="records") if isinstance(raw, pd.DataFrame) else list(raw)
        model_instances = self._validate(rows)

        # Convert models to plain dicts and build a normalized DataFrame
        df = pd.DataFrame([m.model_dump() for m in model_instances])

        return df

    @classmethod
    def from_source(cls, source: str, **kwargs: Any) -> pd.DataFrame:
        """Factory to dispatch registered subclass by `source`.

        Args:
            source: The source identifier (e.g., 'nyc', 'sf', 'boston')
            **kwargs: Arguments passed to the loader's __init__

        Returns:
            Loaded and validated DataFrame of locations

        Example:
            locations = UniverseLoader.from_source('nyc')
        """
        key = str(source).lower()
        try:
            loader_cls = cls._REGISTRY[key]
        except KeyError as e:
            raise ValueError(
                f"Unknown source '{source}'. "
                f"Known sources: {sorted(cls._REGISTRY.keys())}"
            ) from e

        loader = loader_cls(**kwargs)
        return loader.load()

    def _validate(self, rows:Iterable) -> None:
        # Validates the model against a Location
        try: 
            models: list[Location] = [Location.model_validate(r, strict=False, by_name=True) for r in rows]
        except ValidationError as e:
            logger.error(
                'Universe validation failed',
                extra={"source": self.SOURCE, "error_count": len(e.errors())}
            )
            raise DataValidationError(self.SOURCE, e.errors(), original=e)
        
        return models

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
    def _load_pipeline(self, **kwargs) -> Iterable[tuple[str, Callable, list, dict[str, any]]]:
        ...

    # def _to_database(self, df: pd.DataFrame | None = None) -> None:
    #     """Save universe locations to database.

    #     Default implementation does nothing.
    #     Subclasses can override to implement database persistence.
    #     """
    #     if df is None or df.empty:
    #         return

    #     db_path = os.getenv("DDB_PATH")
    #     if db_path is None:
    #         db_path = Path(__file__).resolve().parents[3] / "data" / "core.ddb"
    #     else:
    #         db_path = Path(db_path)

    #     db_path.parent.mkdir(parents=True, exist_ok=True)

    #     try:
    #         import duckdb
    #     except ImportError:
    #         logger.info("DuckDB not installed; skipping persistence for '%s'.", self.SOURCE)
    #         return

    #     prepared = df.copy()

    #     def _is_missing(value: Any) -> bool:
    #         if value is None:
    #             return True
    #         try:
    #             result = pd.isna(value)
    #         except (TypeError, ValueError):
    #             return False
    #         return bool(result) if isinstance(result, (bool, int)) else False

    #     def _first_non_null(series: pd.Series) -> Any:
    #         for value in series:
    #             if _is_missing(value):
    #                 continue
    #             return value
    #         return None

    #     for column in prepared.columns:
    #         sample = _first_non_null(prepared[column])
    #         if isinstance(sample, BaseGeometry):
    #             prepared[column] = prepared[column].apply(
    #                 lambda geom: geom.wkb_hex if isinstance(geom, BaseGeometry) else None
    #             )
    #         elif isinstance(sample, (list, dict)):
    #             prepared[column] = prepared[column].apply(
    #                 lambda value: (
    #                     json.dumps(value)
    #                     if not _is_missing(value)
    #                     else None
    #                 )
    #             )

    #     suffix = re.sub(r"[^0-9a-zA-Z_]", "_", str(self.SOURCE).lower())
    #     if not suffix:
    #         suffix = "default"
    #     if suffix[0].isdigit():
    #         suffix = f"u_{suffix}"
    #     table_name = f"universe_{suffix}"

    #     try:
    #         with duckdb.connect(str(db_path)) as conn:
    #             conn.register("universe_df", prepared)
    #             conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    #             conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM universe_df")
    #             conn.unregister("universe_df")
    #     except duckdb.Error as exc:  # type: ignore[attr-defined]
    #         logger.error("Failed to persist universe '%s' to %s: %s", self.SOURCE, db_path, exc)
