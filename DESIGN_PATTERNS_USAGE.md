# Data Loader Design Patterns - Usage Guide

This guide demonstrates how to use the design patterns implemented in the st_preprocessing package.

## Table of Contents

1. [Registry Pattern (ProjectLoader, FeatureLoader, UniverseLoader)](#registry-pattern)
2. [Composition Pattern (Transformers)](#composition-pattern)
3. [Strategy Pattern (ImageryLoader)](#strategy-pattern)
4. [Creating New Loaders](#creating-new-loaders)

---

## Registry Pattern

The Registry pattern is used by `ProjectLoader`, `FeatureLoader`, and `UniverseLoader` to automatically register data sources and provide a factory method for instantiation.

### Basic Usage

```python
from st_preprocessing.projects.project_loader import ProjectLoader

# Option 1: Use the factory method (recommended)
projects = ProjectLoader.from_source('nyc')

# Option 2: Instantiate directly
from st_preprocessing.projects.sources.nyc import ProjectLoaderNYC
loader = ProjectLoaderNYC()
projects = loader.load()
```

### With Custom Parameters

```python
from pathlib import Path

# Pass parameters through the factory
projects = ProjectLoader.from_source(
    'nyc',
    data_path=Path('./my_data'),
    project_type='ALL'  # Load all project types, not just CAPITAL RECONSTRUCTION
)
```

### Creating a New Source Loader

```python
from st_preprocessing.projects.project_loader import ProjectLoader
import geopandas as gpd

class ProjectLoaderBoston(ProjectLoader):
    SOURCE = 'boston'  # Automatically registers as 'boston'

    def __init__(self, data_path='./boston_data'):
        self.data_path = data_path
        super().__init__()

    def _load_raw(self) -> gpd.GeoDataFrame:
        # Your loading logic here
        return gpd.read_file(self.data_path)

# Now you can use it via the registry:
projects = ProjectLoader.from_source('boston')
```

---

## Composition Pattern (Transformers)

Transformers allow you to build data processing pipelines by composing small, reusable transformation steps.

### Available Transformers

```python
from st_preprocessing.transformers import (
    WKTGeometryTransformer,      # Convert WKT to geometries
    DateTimeTransformer,          # Parse date strings
    ColumnFilterTransformer,      # Filter rows
    ColumnSelectorTransformer,    # Select specific columns
    ColumnRenameTransformer,      # Rename columns
    DropNATransformer,            # Drop rows with missing values
    CRSTransformer,               # Transform coordinate systems
    CustomFunctionTransformer,    # Apply custom functions
    CompositeTransformer,         # Chain multiple transformers
)
```

### Basic Usage

```python
from st_preprocessing.transformers import (
    WKTGeometryTransformer,
    DateTimeTransformer,
    ColumnFilterTransformer,
)

# Create a transformation pipeline
transformers = [
    WKTGeometryTransformer(geom_column='the_geom', crs='EPSG:4326'),
    DateTimeTransformer(columns=['start_date', 'end_date']),
    ColumnFilterTransformer(status='Active'),
]

# Use with any loader
from st_preprocessing.projects.sources.nyc import ProjectLoaderNYC

loader = ProjectLoaderNYC(transformers=transformers)
projects = loader.load()
```

### Advanced Filtering

```python
from st_preprocessing.transformers import ColumnFilterTransformer

# Simple equality filter
filter1 = ColumnFilterTransformer(ProjectType='CAPITAL RECONSTRUCTION')

# Callable filter (lambda or function)
filter2 = ColumnFilterTransformer(
    ProjectCost=lambda x: x > 1000000
)

# Multiple filters
filter3 = ColumnFilterTransformer(
    ProjectType='CAPITAL RECONSTRUCTION',
    Status='Active',
    ProjectCost=lambda x: x > 1000000
)
```

### Custom Transformers

```python
from st_preprocessing.transformers import Transformer, CustomFunctionTransformer
import pandas as pd

# Option 1: Use CustomFunctionTransformer
def add_year_column(df):
    df['year'] = df['date'].dt.year
    return df

transformer = CustomFunctionTransformer(add_year_column)

# Option 2: Create a reusable transformer class
class YearExtractorTransformer(Transformer):
    def __init__(self, date_column='date', output_column='year'):
        self.date_column = date_column
        self.output_column = output_column

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.output_column] = data[self.date_column].dt.year
        return data

transformer = YearExtractorTransformer(date_column='ConstructionEndDate')
```

### Composite Transformers

```python
from st_preprocessing.transformers import CompositeTransformer

# Create a reusable pipeline
nyc_cleanup_pipeline = CompositeTransformer([
    WKTGeometryTransformer(),
    DateTimeTransformer(['ConstructionEndDate']),
    ColumnFilterTransformer(ProjectType='CAPITAL RECONSTRUCTION'),
    DropNATransformer(columns=['geometry']),
])

# Use it
loader = ProjectLoaderNYC(transformers=[nyc_cleanup_pipeline])
```

---

## Strategy Pattern (ImageryLoader)

The Strategy pattern allows you to swap out different imagery sources without changing the loader code.

### Loading from Local Files

```python
from st_preprocessing.imagery.imagery_loader import ImageryLoader
from st_preprocessing.imagery.source_strategies import LocalFileSource

# Define your locations
locations = [
    (40.7128, -74.0060),  # NYC
    (40.7589, -73.9851),  # Times Square
]

# Option 1: Direct instantiation
source = LocalFileSource(
    directory='./images',
    file_pattern='{lat}_{lon}',
    file_extension='.png'
)
loader = ImageryLoader(
    source_strategy=source,
    region='manhattan',
    locations=locations
)
images = loader.load()

# Option 2: Convenience method
loader = ImageryLoader.from_local(
    file_dir='./images',
    region='manhattan',
    locations=locations
)
images = loader.load()
```

### Loading from Google Maps API

```python
from st_preprocessing.imagery.imagery_loader import ImageryLoader
from st_preprocessing.imagery.source_strategies import GoogleMapsAPISource
import os

# Create API source
api_source = GoogleMapsAPISource(
    api_key=os.getenv('GOOGLE_MAPS_API_KEY'),
    zoom=18,
    size='640x640',
    maptype='satellite'
)

loader = ImageryLoader.from_api(
    api_source=api_source,
    region='manhattan',
    locations=locations
)
images = loader.load()
```

### Using Caching

```python
from st_preprocessing.imagery.source_strategies import (
    GoogleMapsAPISource,
    CachedSource
)

# Wrap API source with caching
api_source = GoogleMapsAPISource(api_key=os.getenv('GOOGLE_MAPS_API_KEY'))
cached_source = CachedSource(
    source=api_source,
    cache_dir='./image_cache',
    cache_pattern='{lat}_{lon}.png'
)

loader = ImageryLoader(
    source_strategy=cached_source,
    region='manhattan',
    locations=locations
)

# First call fetches from API and caches
images = loader.load()

# Second call loads from cache (much faster!)
images = loader.load()

# Or use the convenience method
loader = ImageryLoader.from_api(
    api_source=api_source,
    region='manhattan',
    locations=locations,
    cache_dir='./image_cache'  # Automatically wraps in CachedSource
)
```

### Creating Custom Imagery Sources

```python
from st_preprocessing.imagery.source_strategies import ImagerySource
from typing import Any, Iterable

class MapboxAPISource(ImagerySource):
    def __init__(self, access_token: str, style: str = 'satellite-v9'):
        self.access_token = access_token
        self.style = style

    def fetch(self, location: tuple[float, float], **kwargs: Any) -> bytes:
        import requests
        lon, lat = location  # Mapbox uses lon, lat order

        url = f"https://api.mapbox.com/styles/v1/mapbox/{self.style}/static/{lon},{lat},15/600x400"
        params = {'access_token': self.access_token}

        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.content

    def fetch_batch(self, locations: Iterable[tuple[float, float]], **kwargs: Any) -> list[bytes]:
        return [self.fetch(loc, **kwargs) for loc in locations]

# Use it
mapbox_source = MapboxAPISource(access_token=os.getenv('MAPBOX_TOKEN'))
loader = ImageryLoader(
    source_strategy=mapbox_source,
    region='manhattan',
    locations=locations
)
```

---

## Creating New Loaders

### Adding a New Project Source

```python
# File: src/st_preprocessing/projects/sources/sf.py

from ..project_loader import ProjectLoader
from ...transformers import WKTGeometryTransformer
import geopandas as gpd
from pathlib import Path

class ProjectLoaderSF(ProjectLoader):
    SOURCE = 'sf'  # Registers as San Francisco

    def __init__(self, data_path: Path = None, transformers=None):
        self.data_path = data_path or Path('./sf_data')

        default_transformers = [
            WKTGeometryTransformer(geom_column='geometry', crs='EPSG:4326'),
        ]

        all_transformers = default_transformers + (transformers or [])
        super().__init__(transformers=all_transformers)

    def _load_raw(self) -> gpd.GeoDataFrame:
        return gpd.read_file(self.data_path / 'sf_projects.geojson')

# Usage
from st_preprocessing.projects.project_loader import ProjectLoader
projects = ProjectLoader.from_source('sf')
```

### Adding a New Feature Source

```python
# File: src/st_preprocessing/features/sources/census.py

from ..feature_loader import FeatureLoader
import pandas as pd

class CensusFeatureLoader(FeatureLoader):
    SOURCE = 'census'

    def __init__(self, year: int = 2020, geography: str = 'tract'):
        self.year = year
        self.geography = geography
        super().__init__()

    def _load_raw(self) -> pd.DataFrame:
        # Your census data loading logic
        import cenpy
        return cenpy.products.ACS(self.year).from_place(geography=self.geography)

# Usage
from st_preprocessing.features.feature_loader import FeatureLoader
census_data = FeatureLoader.from_source('census', year=2020, geography='tract')
```

---

## Complete Example: Building a Data Pipeline

```python
from pathlib import Path
from st_preprocessing.projects.project_loader import ProjectLoader
from st_preprocessing.locations.universe import UniverseLoader
from st_preprocessing.imagery.imagery_loader import ImageryLoader
from st_preprocessing.imagery.source_strategies import GoogleMapsAPISource, CachedSource
from st_preprocessing.transformers import (
    ColumnFilterTransformer,
    DateTimeTransformer,
    DropNATransformer,
)

# 1. Load locations (street universe)
locations = UniverseLoader.from_source('osm', path='streets.osm')

# 2. Load projects with custom filtering
projects = ProjectLoader.from_source(
    'nyc',
    transformers=[
        DateTimeTransformer(['ConstructionEndDate']),
        ColumnFilterTransformer(
            ProjectType='CAPITAL RECONSTRUCTION',
            ConstructionEndDate=lambda x: x.dt.year >= 2020
        ),
        DropNATransformer(columns=['geometry', 'ProjectCost']),
    ]
)

# 3. Load imagery with caching
api_source = GoogleMapsAPISource(api_key='YOUR_KEY')
imagery_loader = ImageryLoader.from_api(
    api_source=api_source,
    region='nyc',
    locations=[(row.geometry.centroid.y, row.geometry.centroid.x)
               for _, row in projects.iterrows()],
    cache_dir='./cache'
)
images = imagery_loader.load()

# 4. Combine everything
projects['image_path'] = images
```

---

## Benefits of These Patterns

1. **Extensibility**: Add new sources by creating a class, no need to modify existing code
2. **Reusability**: Transformers and strategies can be reused across different loaders
3. **Testability**: Each component can be tested independently
4. **Maintainability**: Clear separation of concerns makes code easier to understand
5. **Flexibility**: Mix and match components to build custom pipelines
6. **Type Safety**: Full type hints for better IDE support and error detection
