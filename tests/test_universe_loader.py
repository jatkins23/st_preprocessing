from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1] # This sucks
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


import pandas as pd  # type: ignore
from st_preprocessing.utils.errors import DatasetValidationError
from st_preprocessing.locations.universe import UniverseLoader


@pytest.fixture(autouse=True)
def reset_registry():
    original_registry = UniverseLoader._REGISTRY.copy()
    UniverseLoader._REGISTRY.clear()
    yield
    UniverseLoader._REGISTRY = original_registry


def test_load_returns_normalized_dataframe():
    class DemoLoader(UniverseLoader):
        SOURCE = "demo"

        def _load_raw(self):
            return [
                {"name": "Alpha", "source": "demo"},
                {"name": "Beta", "source": "demo"},
            ]

    loader = DemoLoader()

    frame = loader.load()

    assert isinstance(frame, pd.DataFrame)
    assert frame.to_dict(orient="records") == [
        {"name": "Alpha", "source": "demo"},
        {"name": "Beta", "source": "demo"},
    ]


def test_load_invalid_rows_raise_dataset_validation_error():
    class BrokenLoader(UniverseLoader):
        SOURCE = "broken"

        def _load_raw(self):
            return [
                {"name": "OnlyName"},
            ]

    loader = BrokenLoader()

    with pytest.raises(DatasetValidationError) as excinfo:
        loader.load()

    err = excinfo.value
    assert err.source == "broken"
    assert err.errors


def test_from_source_dispatches_registered_loader():
    call_state = {"loaded": False}

    class TrackingLoader(UniverseLoader):
        SOURCE = "tracking"

        def __init__(self, state):
            self._state = state

        def _load_raw(self):
            self._state["loaded"] = True
            return [{"name": "Tracked", "source": "tracking"}]

    UniverseLoader.from_source("tracking", state=call_state)

    assert call_state["loaded"] is True

