# All of the sources for geospatial imagery
from abc import ABC, abstractmethod

class ImagerySource(ABC):
    name: str
    keyword: str
    server: str
    years: list[int]
    zoomlevels: list = list(range(0, 20))


class NewYork(ImagerySource):
    name = 'nyc'
    keyword = 'New York City', 'City of New York'
    years = list(range(2006, 2024, 2))
    URL_TEMPLATE = (
            "https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/"
            "services/NYC_Orthos_{year}/MapServer"
        )
    
    @abstractmethod
    def _format_base_url(url_template:str, year:int) -> str:
        year_string = year if year != 2020 else '-_2020'
        return url_template.format(year=year_string)
    
    @property
    def server(self):
        self._format_base_url(self.URL_TEMPLATE, self.year)