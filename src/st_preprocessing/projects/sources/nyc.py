# Source data for loading and cleaning capital reconstruction projects from NYC OpenData
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import pandas as pd
import geopandas as gpd
from shapely import from_wkt

from ..project_loader import ProjectLoader
from ...transformers import (
    WKTGeometryTransformer,
    DateTimeTransformer,
    ColumnFilterTransformer,
    ColumnSelectorTransformer,
)

# Configuration
DATA_PATH = Path(str(os.getenv('DATA_PATH', '.')))
OPENNYC_DATA_PATH = DATA_PATH / 'raw' / 'citydata' / 'openNYC'
CORE_FILE_NAME = 'Street_and_Highway_Capital_Reconstruction_Projects_-_Intersection_20250721.csv'

COLUMNS_TO_KEEP = [
    'ProjectID', 'ProjTitle', 'FMSID', 'FMSAgencyID',
    'LeadAgency', 'Managing Agency', 'ProjectDescription',
    'ProjectTypeCode', 'ProjectType', 'ProjectStatus', 'ConstructionFY',
    'DesignStartDate', 'ConstructionEndDate', 'CurrentFunding',
    'ProjectCost', 'OversallScope', 'SafetyScope', 'OtherScope',
    'ProjectJustification', 'OnStreetName', 'FromStreetName',
    'ToStreetName', 'OFTCode', 'DesignFY', 'geometry'
]

DROP_SCOPES = [
    'Wayfindiing', 'Resurfacing', 'Median-Planted Trees',
    'DEP Project', 'Percent for Art', 'GI in Scope', 'Potential GI'
]


class ProjectLoaderNYC(ProjectLoader):
    """Load NYC capital reconstruction projects from OpenData.

    This loader automatically registers itself as 'nyc' in the ProjectLoader registry.

    Args:
        data_path: Path to directory containing NYC OpenData files
        file_name: Name of the CSV file to load
        project_type: Filter for specific project type (default: 'CAPITAL RECONSTRUCTION')
        input_crs: Input coordinate reference system (default: 'EPSG:4326')
        transformers: Optional list of additional transformers to apply

    Example:
        # Using the registry
        projects = ProjectLoader.from_source('nyc')

        # Or instantiate directly
        loader = ProjectLoaderNYC()
        projects = loader.load()

        # With custom filtering
        loader = ProjectLoaderNYC(project_type='ALL')
        projects = loader.load()
    """

    SOURCE = 'nyc'

    def __init__(
        self,
        data_path: Path = OPENNYC_DATA_PATH,
        file_name: str = CORE_FILE_NAME,
        project_type: str = 'CAPITAL RECONSTRUCTION',
        input_crs: str = 'EPSG:4326',
        transformers: Optional[list] = None,
    ):
        """Initialize NYC project loader with configuration."""
        self.data_path = data_path
        self.file_name = file_name
        self.project_type = project_type
        self.input_crs = input_crs

        # Build default transformation pipeline
        default_transformers = [
            WKTGeometryTransformer(geom_column='the_geom', crs=input_crs),
            DateTimeTransformer(columns=['DesignStartDate', 'ConstructionEndDate']),
        ]

        # Add project type filter if specified
        if project_type and project_type != 'ALL':
            default_transformers.append(
                ColumnFilterTransformer(ProjectType=project_type)
            )

        # Add column selector
        default_transformers.append(
            ColumnSelectorTransformer(columns=COLUMNS_TO_KEEP)
        )

        # Combine with any user-provided transformers
        all_transformers = default_transformers + (transformers or [])

        super().__init__(transformers=all_transformers)

    def _load_raw(self) -> pd.DataFrame:
        """Load raw project data from CSV file.

        Returns:
            Raw DataFrame with NYC project data
        """
        source_file_path = self.data_path / self.file_name

        if not source_file_path.exists():
            raise FileNotFoundError(
                f"NYC projects file not found: {source_file_path}\n"
                f"Please ensure DATA_PATH is set correctly in your environment."
            )

        return pd.read_csv(source_file_path)


# Utility function for backward compatibility
def load_standard(path: Path, crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
    """Load a CSV file with WKT geometry column into a GeoDataFrame.

    Args:
        path: Path to CSV file
        crs: Coordinate reference system

    Returns:
        GeoDataFrame with parsed geometries
    """
    df = pd.read_csv(path)
    return gpd.GeoDataFrame(df, geometry=from_wkt(df['the_geom']), crs=crs)
