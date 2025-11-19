from pydantic import BaseModel, ConfigDict, PastDate 
from datetime import date
from typing import Optional, Union

from shapely.geometry import Point, LineString, Polygon

class CityProject(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_id: int
    overall_scope: list[str]
    project_scope: list[str]
    safety_scope: list[str]
    start_time: Optional[PastDate]
    end_time: Optional[date]
    geometry: Union[Point, LineString, Polygon]