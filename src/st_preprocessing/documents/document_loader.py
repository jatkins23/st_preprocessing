# Module to load and handle documents from various source
# - 1) collect
# - 2) Classify
# - 3) Geolocate
#       3.1) Streetfind: find sterete
#       3.2) Geocode: assign streets to coordinates (maybe not necessary)
# - 4) Align: align documents with locations by street names. But also maybe align docuemnts with projects by name/location?


from abc import ABC
from pydantic import BaseModel, FilePath

class DocumentPage(BaseModel):
    document_page_id: int
    document_file_id: int
    document_collection_id: int
    page_number: int
    file_path: FilePath    

class DocumentFile(BaseModel):
    document_file_id: int
    document_collection_id: int
    pages: list[DocumentPage]


class DocumentCollection(BaseModel):
    document_collection_id: int
    document_file_id: int


class DocumentsLoader(ABC):
    pass