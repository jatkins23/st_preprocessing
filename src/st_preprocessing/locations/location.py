from pathlib import Path

from typing import Optional, Iterable
from shapely.geometry import Point
from pydantic import BaseModel, ConfigDict

class Location(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    location_id: int
    street1: str
    street2: Optional[str]
    additional_streets: Optional[list[str]]
    street_count: Optional[int]
    original_nodeids: Optional[list[int]]
    geometry: Point
