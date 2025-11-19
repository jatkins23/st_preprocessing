# Data Loader Architecture

## Overview

The data loading system uses a **clean, layered architecture** with clear separation of concerns and no circular dependencies.

## Core Principles

1. **Single Responsibility**: Each class has one clear purpose
2. **Dependency Injection**: Components are composed, not tightly coupled
3. **Open/Closed**: Easy to extend with new loaders without modifying existing code
4. **Mixin Pattern**: Optional features (like pipelines) are added via mixins

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                        DataLoader (ABC)                      │
│  - Minimal base class with MODALITY registry                │
│  - Factory method: load(modality, source)                   │
│  - Abstract: _load_raw()                                    │
│  - Utility: to_database()                                   │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ UniverseLoader  │  │ FeatureLoader   │  │ ProjectLoader   │
│ MODALITY='universe'  MODALITY='features'  MODALITY='projects'
│ + from_source() │  │ + from_source() │  │ + from_source() │
│ + _REGISTRY     │  │ + _REGISTRY     │  │ + _REGISTRY     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  LIONLoader     │  │ CensusLoader    │  │  NYCLoader      │
│  SOURCE='lion'  │  │ SOURCE='census' │  │  SOURCE='nyc'   │
│ + PipelineMixin │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## File Structure

```
src/st_preprocessing/
├── data_loader.py           # Base DataLoader class
├── pipeline_mixin.py        # Optional pipeline execution mixin
├── transformers.py          # Data transformation utilities
│
├── locations/
│   ├── universe.py          # UniverseLoader (modality)
│   └── sources/
│       ├── lion.py          # LIONLoader (uses PipelineMixin)
│       └── osm.py           # OSMLoader
│
├── features/
│   ├── feature_loader.py    # FeatureLoader (modality)
│   └── sources/             # Feature source implementations
│
├── projects/
│   ├── project_loader.py    # ProjectLoader (modality)
│   └── sources/
│       └── nyc.py           # NYC project loader
│
└── imagery/
    ├── imagery_loader.py    # ImageryLoader (modality)
    └── source_strategies.py # Strategy pattern implementations
```

---

## Component Responsibilities

### 1. DataLoader (Base ABC)

**File**: `data_loader.py`

**Responsibilities**:
- Define minimal interface all loaders must implement
- Maintain MODALITY registry for modality loaders
- Provide factory method `load(modality, source, **kwargs)`
- Provide database utility `to_database(df, table_name, schema_name)`

**Abstract Methods**:
- `_load_raw()`: Must be implemented by subclasses

**Concrete Methods**:
- `load()`: Factory dispatch by modality
- `to_database()`: Save DataFrame to DuckDB
- `_validate()`: Default no-op (can be overridden)

### 2. Modality Loaders (UniverseLoader, FeatureLoader, etc.)

**Files**: `universe.py`, `feature_loader.py`, `project_loader.py`, `imagery_loader.py`

**Responsibilities**:
- Define `MODALITY` class variable (auto-registers with DataLoader)
- Maintain SOURCE registry for source-specific loaders
- Provide `from_source(source, **kwargs)` factory method
- Implement modality-specific loading logic

**Key Pattern**:
```python
class UniverseLoader(DataLoader):
    MODALITY = "universe"  # Auto-registers with DataLoader
    SOURCE: ClassVar[str]  # Subclasses set this
    _REGISTRY = {}         # SOURCE → loader class mapping

    @classmethod
    def from_source(cls, source: str, **kwargs):
        loader_cls = cls._REGISTRY[source.lower()]
        loader = loader_cls(**kwargs)
        return loader.load()
```

### 3. Source Loaders (LIONLoader, NYCLoader, etc.)

**Files**: `locations/sources/lion.py`, `projects/sources/nyc.py`, etc.

**Responsibilities**:
- Define `SOURCE` class variable (auto-registers with modality)
- Implement `_load_raw()` to return data
- Optionally use `PipelineMixin` for multi-step processing

**Key Pattern**:
```python
class LIONLoader(UniverseLoader, PipelineMixin):
    SOURCE = "lion"  # Auto-registers with UniverseLoader

    def _load_raw(self):
        return self._execute_pipeline()  # From PipelineMixin

    def _load_pipeline(self):
        return [
            ('Step 1', self.method1, [], {}),
            ('Step 2', self.method2, [], {}),
        ]
```

### 4. PipelineMixin (Optional)

**File**: `pipeline_mixin.py`

**Responsibilities**:
- Execute multi-step pipelines with progress reporting
- Handle errors per pipeline step
- Format and print step results

**Usage**:
```python
class MyLoader(SomeModality, PipelineMixin):
    def _load_pipeline(self):
        return [
            ('Load Data', self.load_data, [], {}),
            ('Transform', self.transform, [], {'param': value}),
        ]

    def _load_raw(self):
        return self._execute_pipeline(progress=True)
```

---

## Key Design Decisions

### ✅ What We Fixed

1. **Removed Circular Logic**
   - **Before**: DataLoader had pipeline logic, but not all modalities use pipelines
   - **After**: Pipeline logic moved to optional PipelineMixin

2. **Clearer Abstractions**
   - **Before**: `_load_raw()` had default implementation calling `_load_pipeline()`
   - **After**: `_load_raw()` is purely abstract; pipelines are opt-in via mixin

3. **Simplified Dependencies**
   - **Before**: Modality loaders imported DataLoader, DataLoader had modality-specific code
   - **After**: Clean one-way dependency: DataLoader ← Modality ← Source

4. **Better Separation of Concerns**
   - **Before**: Database logic mixed with loading logic
   - **After**: `to_database()` is standalone utility method

### 🎯 Current Architecture Benefits

1. **No Circular Dependencies**: Clean import hierarchy
2. **Easy to Extend**: Add new loaders by defining `SOURCE`/`MODALITY`
3. **Testable**: Each component can be tested independently
4. **Flexible**: Mix and match features (PipelineMixin is optional)
5. **Type-Safe**: Each modality can return different types (DataFrame, GeoDataFrame, list)

---

## Usage Examples

### Basic Usage

```python
from st_preprocessing.data_loader import DataLoader

# Option 1: Via DataLoader factory
universe = DataLoader.load(modality='universe', source='lion')

# Option 2: Via modality loader
from st_preprocessing.locations import UniverseLoader
universe = UniverseLoader.from_source('lion')

# Option 3: Direct instantiation
from st_preprocessing.locations.sources.lion import LIONLoader
loader = LIONLoader()
universe = loader.load()
```

### Saving to Database

```python
from st_preprocessing.data_loader import DataLoader

# Load data
universe = DataLoader.load(modality='universe', source='lion')

# Save to database
DataLoader.to_database(
    df=universe,
    table_name='locations',
    schema_name='raw'
)
# Saves to: raw.locations
```

### Creating New Loaders

#### Simple Loader (No Pipeline)

```python
from st_preprocessing.features.feature_loader import FeatureLoader
import pandas as pd

class TrafficLoader(FeatureLoader):
    SOURCE = 'traffic'  # Auto-registers

    def _load_raw(self):
        # Simple data loading
        return pd.read_csv('traffic.csv')
```

#### Pipeline Loader (Complex)

```python
from st_preprocessing.projects.project_loader import ProjectLoader
from st_preprocessing.pipeline_mixin import PipelineMixin
import geopandas as gpd

class SFProjectLoader(ProjectLoader, PipelineMixin):
    SOURCE = 'sf'  # Auto-registers

    def _load_raw(self):
        return self._execute_pipeline(progress=True)

    def _load_pipeline(self):
        return [
            ('Fetch API Data', self.fetch_api, [], {}),
            ('Parse GeoJSON', self.parse_geojson, [], {}),
            ('Clean Data', self.clean, [], {}),
        ]

    def fetch_api(self):
        # Fetch from API
        ...

    def parse_geojson(self):
        # Parse response
        ...

    def clean(self):
        # Clean data
        ...
```

---

## Migration Notes

If you have old code that relies on the previous architecture:

### Before
```python
# Old: DataLoader._to_database() was private
loaded_df = modality_cls.from_source(source, **kwargs)
if 'table_name' in kwargs:
    cls._to_database(df=loaded_df, table_name=kwargs['table_name'])
```

### After
```python
# New: Use public to_database() method
loaded_df = modality_cls.from_source(source, **kwargs)
DataLoader.to_database(df=loaded_df, table_name='my_table')
```

---

## Summary

The new architecture is:
- **Simple**: Clear responsibilities, minimal abstractions
- **Flexible**: Mix-ins for optional features
- **Extensible**: Easy to add new loaders
- **Clean**: No circular dependencies or tight coupling
- **Type-Safe**: Each modality can define its own return types

This design follows SOLID principles and makes the codebase easier to understand, test, and maintain.
