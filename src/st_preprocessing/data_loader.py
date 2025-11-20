from __future__ import annotations

from typing import Any, ClassVar, Type, Optional
import logging

import pandas as pd
import geopandas as gpd

from abc import ABC, abstractmethod
from .db.db import duckdb_connection

logger = logging.getLogger('DataLoader')


class DataLoader(ABC):
    """Minimal abstract base class for all data loaders.

    Three-level hierarchy:
    1. DataLoader (this class) - Minimal ABC with MODALITY registry
    2. Modality loaders (UniverseLoader, FeatureLoader, etc.) - Each with SOURCE registry
    3. Specific implementations (LIONLoader, etc.) - Register with their modality

    Usage:
        # Option 1: Use the specific modality loader directly
        data = UniverseLoader.from_source('lion')

        # Option 2: Use DataLoader.load() to dispatch by modality
        data = DataLoader.load(modality='universe', source='lion')
    """

    # Unique key for each modality subclass (e.g., 'universe', 'features', 'projects', 'imagery')
    MODALITY: ClassVar[str]

    # Global registry of modality loaders
    _MODALITY_REGISTRY: ClassVar[dict[str, Type['DataLoader']]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register modality loaders when they're defined."""
        super().__init_subclass__(**kwargs)

        # Auto-register modality loaders that DIRECTLY DEFINE a MODALITY string
        # (not inherited from parent class)
        if "MODALITY" in cls.__dict__:  # Only if defined on this class, not inherited
            modality = cls.MODALITY
            key = str(modality).lower()
            if key in DataLoader._MODALITY_REGISTRY and DataLoader._MODALITY_REGISTRY[key] is not cls:
                raise RuntimeError(f"Duplicate loader MODALITY '{key}' for {cls.__name__}")
            DataLoader._MODALITY_REGISTRY[key] = cls
            logger.debug(f"Registered DataLoader modality: {cls.__name__} as '{key}'")

    @classmethod
    def load(cls, modality: str, source: str, **kwargs: Any) -> Any:
        """Factory method to load data by modality and source.

        Args:
            modality: The data modality (e.g., 'universe', 'features', 'projects', 'imagery')
            source: The specific source within that modality (e.g., 'lion', 'census', 'nyc')
            **kwargs: Arguments passed to the source loader's __init__

        Returns:
            Loaded data from the specified modality and source

        Example:
            # Load LION universe data
            universe = DataLoader.load(modality='universe', source='lion')

            # Load census features
            features = DataLoader.load(modality='features', source='census', year=2020)
        """
        key = str(modality).lower()
        try:
            modality_cls = cls._MODALITY_REGISTRY[key]
        except KeyError as e:
            raise ValueError(
                f"Unknown modality '{modality}'. "
                f"Known modalities: {sorted(cls._MODALITY_REGISTRY.keys())}"
            ) from e

        # Dispatch to the modality loader's from_source method
        if not hasattr(modality_cls, 'from_source'):
            raise AttributeError(
                f"Modality loader {modality_cls.__name__} does not implement from_source()"
            )

        return modality_cls.from_source(source, **kwargs)

    @abstractmethod
    def _load_raw(self):
        """Load raw data from source.

        Subclasses must implement this to return raw data in their native format.
        For example:
        - UniverseLoader: Returns GeoDataFrame with location data
        - FeatureLoader: Returns DataFrame or GeoDataFrame with features
        - ProjectLoader: Returns GeoDataFrame with project data
        - ImageryLoader: Returns list of image data

        Returns:
            Data in format specific to the modality
        """
        ...

    def _validate(self):
        """Validate loaded data.

        Default implementation does no validation.
        Subclasses can override to add validation logic.
        """
        pass

    @classmethod
    def to_database(
        cls,
        df: pd.DataFrame | gpd.GeoDataFrame,
        table_name: str,
        schema_name: Optional[str] = None
    ) -> str:
        """Save DataFrame to DuckDB database.

        Args:
            df: DataFrame or GeoDataFrame to save
            table_name: Name of the table
            schema_name: Optional schema name

        Returns:
            Full table name (schema.table or just table)
        """
        with duckdb_connection() as db_con:
            # Create schema if specified
            if schema_name:
                db_con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name};")
                full_name = f"{schema_name}.{table_name}"
            else:
                full_name = table_name

            # Handle GeoDataFrame geometry column
            geom_col = None
            if isinstance(df, gpd.GeoDataFrame) and df.geometry is not None:
                geom_col = df.geometry.name
                # Convert to regular DataFrame and replace geometry with WKT
                df_to_save = pd.DataFrame(df.drop(columns=[geom_col]))
                df_to_save[geom_col] = df.geometry.apply(
                    lambda geom: geom.wkt if geom is not None else None
                )
            else:
                df_to_save = df.copy()

            # Register DataFrame and create table
            db_con.register("_tmp_gdf", df_to_save)
            try:
                db_con.execute(f"DROP TABLE IF EXISTS {full_name};")

                # If GeoDataFrame, convert WKT to geometry in the table
                if geom_col:
                    # Get all non-geometry columns
                    other_cols = [col for col in df_to_save.columns if col != geom_col]
                    cols_select = ", ".join(other_cols)

                    db_con.execute(f"""
                        CREATE TABLE {full_name} AS
                        SELECT {cols_select},
                               ST_GeomFromText({geom_col}) AS {geom_col}
                        FROM _tmp_gdf
                    """)
                else:
                    db_con.execute(f"CREATE TABLE {full_name} AS SELECT * FROM _tmp_gdf;")

                logger.info(f"Saved {len(df)} rows to {full_name}")
            finally:
                db_con.unregister("_tmp_gdf")

        return full_name
