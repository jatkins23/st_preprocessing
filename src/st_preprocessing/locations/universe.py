# Abstract Class for a LocationSet, which contains necessary 
# -- Maybe rename to 'Region'
from __future__ import annotations

from typing import Any, ClassVar, Type, Iterable, Mapping, Callable
import logging
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
from .location_geometry import LocationGeometry
from ..data_loader import DataLoader
from ..db.db import duckdb_connection, load_wkt_gdf

logger = logging.getLogger('UniverseLoader')

class Universe(BaseModel, arbitrary_types_allowed=True):
    # TODO: turn locations and location_geometries into Iterables. Make properties out of location_gdf and location_geometries_gdf
    source:                 str # nyc
    name:                   str | None = None # If name isn't given, make it source
    locations:              gpd.GeoDataFrame | None = None # Iterable[Location]
    location_geometries:    pd.DataFrame | None = None # Iterable[LocationGeometry]
    location_image_paths:   Path | None = None

    def model_post_init(self, __context):
        if not self.name:
            self.name = self.source

    # # TODO: turn into properties
    # def get_locations(self):
    #     with duckdb_connection() as db_con:
    #         locations_gdf = load_wkt_gdf(db_con, table_name=f'{self.name}.locations', geom_col='geometry')
    #         locations: list[Location] = [Location.model_validate(r, strict=False, by_name=True) for r in locations_gdf.to_dict(orient='rows')]
    #     return locations_gdf, locations
    
    # def get_location_geometries(self):
    #     with duckdb_connection() as db_con:
    #         try:
    #             df = db_con.execute(f"SELECT * FROM {self.name}.location_geometries;").df()
    #         except Exception as e:
    #             logger.ERROR(e)
    #             return None
    #     return df
            
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

    @classmethod
    def from_source(cls, source: str, persist: bool = True, **kwargs: Any) -> pd.DataFrame:
        """Factory to load universe from source.

        Args:
            source: The source identifier (e.g., 'lion', 'sf', 'boston')
            persist: Whether to perist to database (default: True)
            **kwargs: Arguments passed to load methods

        Returns:
            Universe object

        Example:
            locations = UniverseLoader.from_source('lion')
        """
        key = str(source).lower()
        try:
            loader_cls = cls._REGISTRY[key]
        except KeyError as e:
            raise ValueError(
                f"Unknown source '{source}'. "
                f"Known sources: {sorted(cls._REGISTRY.keys())}"
            ) from e
        
        loader = loader_cls()

        if kwargs.get('with_geometries', True):
            print('loading with geometries')
            universe = loader.load_with_geometries(**kwargs)
        else:
            print('loading without geometries')
            universe = loader.load(**kwargs)

        # Optionally persist
        if persist:
            loader.persist(universe)

        return universe

    def load(self, **kwargs) -> Universe:
        """Load universe data from source.
        Returns clean Universe object without side effects.
        """
        # 1. Load raw data
        raw_gdf = self._load_raw()

        # 2. Validate
        locations = self._validate_locations(raw_gdf)

        # 3. Create Universe
        uni = Universe(
            source=self.SOURCE,
            name=kwargs.get('universe_name', self.SOURCE),
            locations=locations
        )

        return uni

    def load_with_geometries(
        self,
        tile_width: int = 3,
        zlevel: int = 20,
        **kwargs
    ) -> Universe:
        """Load universe with location geometries."""
        uni = self.load(**kwargs)

        # Create location geometries
        location_geoms = self._create_location_geometries(
            uni.locations,
            tile_width=tile_width,
            zlevel=zlevel
        )

        uni.location_geometries = location_geoms
        
        return uni

    def persist(
            self,
            universe: Universe,
            write_locations: bool = True,
            write_geometries: bool = True
    ) -> None:
        """
            Persist universe to database.
            Separated from loading for better testability and flexibility.
        """
        if write_locations and universe.locations is not None:
            super().to_database(
                df=universe.locations,
                table_name='locations',
                schema_name=universe.name
            )

        if write_geometries and universe.location_geometries is not None:
            super().to_database(
                df=universe.location_geometries,
                table_name='location_geometries',
                schema_name=universe.name
            )

    @classmethod
    def _universe_exists_in_db(cls, universe_name: str) -> bool:
        """Check if a universe exists in the database.

        Args:
            universe_name: Name of the universe to check

        Returns:
            True if the universe exists in the database, False otherwise
        """
        try:
            with duckdb_connection() as db_con:
                # Check if schema exists
                result = db_con.execute(f"""
                    SELECT COUNT(*)
                    FROM information_schema.schemata
                    WHERE schema_name = '{universe_name}'
                """).fetchone()

                if result[0] == 0:
                    return False

                # Check if locations table exists
                result = db_con.execute(f"""
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_schema = '{universe_name}'
                    AND table_name = 'locations'
                """).fetchone()

                return result[0] > 0
        except Exception as e:
            logger.error(f"Error checking if universe exists: {e}")
            return False

    @classmethod
    def from_db(
        cls,
        universe_name: str,
        source: str | None = None,
        required: bool = True,
        **kwargs: Any
    ) -> Universe | None:
        """Load universe from database by name.

        Args:
            universe_name: Name of the universe in the database
            source: Optional source identifier (defaults to universe_name if in registry)
            required: If True, raise error if not found. If False, return None (default: True)

        Returns:
            Universe object or None if not found and required=False

        Raises:
            ValueError: If universe not found and required=True

        Example:
            # Load from database (error if not found)
            universe = UniverseLoader.from_db('lion')

            # Load from database (return None if not found)
            universe = UniverseLoader.from_db('lion', required=False)
        """
        if not cls._universe_exists_in_db(universe_name):
            if required:
                raise ValueError(
                    f"Universe '{universe_name}' not found in database. "
                    f"Use UniverseLoader.from_source() to load from source first."
                )
            else:
                logger.info(f"Universe '{universe_name}' not found in database")
                return None

        return cls.from_database(universe_name, source=source)
        
    @classmethod
    def from_database(cls, universe_name: str, source: str) -> Universe:
        """Load universe from database (internal method).

        Args:
            universe_name: Name of the universe in the database
            source: Optional source identifier

        Returns:
            Universe object with locations and location_geometries

        Note:
            Use from_db() instead for better error handling.
        """
        if source and source.lower() not in cls._REGISTRY.keys():
            logger.warning(
                f"Source '{source}' not in registry. "
                f"Known sources: {sorted(cls._REGISTRY.keys())}"
            )

        with duckdb_connection() as db_con:
            # Load locations
            # locations_gdf = load_wkt_gdf(
            #     db_con,
            #     table_name='locations',
            #     table_schema=universe_name,
            #     geom_col='geometry'
            # )
            locations_gdf = gpd.GeoDataFrame(db_con.execute(f'SELECT * FROM {universe_name}.locations;').df())
            logger.info(f"Loaded {len(locations_gdf)} locations from database")

            # Load location_geometries
            try:
                location_geoms_df = db_con.execute(
                    f"SELECT * FROM {universe_name}.location_geometries;"
                ).df()
                logger.info(f"Loaded {len(location_geoms_df)} location geometries from database")
            except Exception as e:
                logger.warning(f"Could not load location_geometries from {universe_name}.location_geometries: {e}")
                location_geoms_df = None

        universe = Universe(
            source=source,
            name=universe_name,
            locations=locations_gdf,
            location_geometries=location_geoms_df
        )

        logger.info(
            f"Loaded universe '{universe_name}' with {len(universe.locations)} locations "
            f"and {len(universe.location_geometries) if universe.location_geometries is not None else 0} geometries"
        )

        return universe

    def _validate_df_to_models(self, raw_df: gpd.GeoDataFrame|pd.DataFrame, model: BaseModel) -> Iterable[BaseModel]:
        """Validate dataframe with given pydantic model."""
        rows = raw_df.to_dict(orient='records')
        try:
            location_models = [
                model.model_validate(r, strict=False)
                for r in rows
            ]
            return location_models
        except ValidationError as e:
            logger.error(
                'Validation for df of {model.__name__} failed',
                extra={'source': self.SOURCE, 'error_count': len(e.errors())}
            )
            raise DataValidationError(self.SOURCE, e.errors(), original=e)

    def _validate_locations(self, raw_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Validate locations against Location model."""
        # Validate model
        location_models = self._validate_df_to_models(raw_gdf, Location)

        # Return as GeoDataFrame
        validated_data = [m.model_dump() for m in location_models]
        return gpd.GeoDataFrame(validated_data, crs=raw_gdf.crs)

    def _create_location_geometries(
        self,
        locations_gdf: gpd.GeoDataFrame,
        tile_width: int,
        zlevel: int
    ) -> pd.DataFrame:
        """Create LocationGeometry objects from locations."""
        # Ensure WGS84 for centroid extraction
        if locations_gdf.crs != 'EPSG:4326':
            locations_gdf = locations_gdf.to_crs('EPSG:4326')
        
        location_models = self._validate_df_to_models(locations_gdf, Location) # This can be handled again with the universe @property
        location_geom_instances = [
            LocationGeometry(
                location_id=loc.location_id,
                centroid=(loc.geometry.x, loc.geometry.y),
                tile_width=tile_width,
                zlevel=zlevel,
                proj_crs=self.CRS
            )
            for loc in location_models
        ]
        location_geoms = pd.DataFrame([lg.model_dump() for lg in location_geom_instances])

        return location_geoms
    
    @abstractmethod
    def _load_pipeline(self, **kwargs) -> Iterable[tuple[str, Callable, list, dict[str, any]]]:
        ...

    # @staticmethod
    # def interpret_boundary(universe_boundary:str|Path|Polygon|gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    #     if isinstance(universe_boundary, (Polygon, gpd.GeoDataFrame)):
    #         bounds=universe_boundary
    #     elif isinstance(universe_boundary, str) and os.path.exists(universe_boundary):
    #         bounds = gpd.read_file(Path(str(universe_boundary)))
    #     elif str(universe_boundary) in ['nyc','all']:
    #         bounds = None
    #     else: # Geocodable address
    #         bounds = ox.geocoder.geocode_to_gdf(str(universe_boundary))

    #     return bounds

    # def clip_by_boundary(self, universe_boundary):
    #     bounds = self.interpret_boundary(universe_boundary)
    #     if bounds is not None:
    #         locations_clipped = self.locations.clip(
    #             bounds.to_crs(self.locations.crs)
    #         )
    #     else:
    #         locations_clipped = self.locations
    # 
    #     return locations_clipped
