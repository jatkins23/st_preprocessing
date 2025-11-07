from pydantic import BaseModel
from abc import ABC

from locations.universe import UniverseLoader, Universe
from projects.project_loader import ProjectLoader
from documents.document_loader import DocumentLoader

class Pipeline(ABC): # pass: config for connecting to db
    name: str

    ul = UniverseLoader()
    self.locations = ul.from_source('LION')

    dl = DocumentLoader()
    self.documents = dl.form_source('NYC')

    _REGISTRY = []
    
    @classmethod
    def from_nyc(cls, outdir: PathLike=None) -> Type('Pipeline'):
        return cls(

        )
        print(source)

if __name__ == '__main__':
    nyc_pipe = Pipeline.from_nyc()
    nyc_pipe()

