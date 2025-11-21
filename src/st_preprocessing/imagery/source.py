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
    keyword = ('New York City', 'City of New York')
    years = list(range(2006, 2024, 2))
    URL_TEMPLATE = (
        "https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/"
        "services/NYC_Orthos_{year}/MapServer/tile/{{z}}/{{y}}/{{x}}"
    )

    def __init__(self, year: int):
        """Initialize NewYork imagery source for a specific year.

        Args:
            year: Year of orthophotos to use (must be in years list)
        """
        if year not in self.years:
            raise ValueError(
                f"Year {year} not available. Available years: {self.years}"
            )
        self.year = year

    @staticmethod
    def _format_base_url(url_template: str, year: int) -> str:
        """Format the URL template with the year string.

        Args:
            url_template: URL template with {year} placeholder
            year: Year to format

        Returns:
            Formatted URL string
        """
        year_string = str(year) if year != 2020 else '-_2020'
        return url_template.format(year=year_string)

    @property
    def server(self) -> str:
        """Get the tile server URL for this year."""
        return self._format_base_url(self.URL_TEMPLATE, self.year)

    def get_tile_url(self) -> str:
        """Get the formatted tile URL with z/y/x placeholders.

        Returns:
            URL string ready for use with TileStitchingSource
        """
        return self.server