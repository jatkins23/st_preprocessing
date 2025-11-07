# Load imagery either from a file
from typing import Iterable

class ImageryLoader:
    def __init__(self, region: str):
        self.region=region

    @classmethod
    def from_local(cls, file_dir):
        pass

    @classmethod
    def from_api(cls, locations: Iterable[tuple[float, float]], source:str):
        pass