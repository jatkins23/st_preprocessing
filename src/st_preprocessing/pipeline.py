# from pydantic import BaseModel
# from abc import ABC

# from locations.universe import UniverseLoader, Universe
# from projects.project_loader import ProjectLoader
# from documents.document_loader import DocumentLoader

# class Pipeline(ABC): # pass: config for connecting to db
#     name: str

#     ul = UniverseLoader()
#     self.locations = ul.from_source('LION')

#     dl = DocumentLoader()
#     self.documents = dl.form_source('NYC')

#     _REGISTRY = []
    
#     @classmethod
#     def from_nyc(cls, outdir: PathLike=None) -> Type('Pipeline'):
#         return cls(

#         )
#         print(source)

# if __name__ == '__main__':
#     nyc_pipe = Pipeline.from_nyc()
#     nyc_pipe()


nyc_universe = UniverseLoader.from_source('lion')
documents = DocumentLoader.from_place('nyc')
projects = ProjectLoader.from_place('nyc')
features = FeatureLoader.from_place('nyc')
imagery = ImageryLoader.from_place('nyc')

nyc_universe.add_modality(imagery, 'imagery') # creates nyc_universe's imagery argument to the
nyc_universe.add_modality(documents, 'documents')
nyc_universe.add_modality(projects, 'projects')
nyc_universe.add_modality(features, 'features')


nyc_universe = NYCUniverse()

class NYCUniverse(Universe):
    def from_subset():
        pass

    def from_region():
        pass

    imagery = ImageryLoader.from_place('nyc')
    projects = ProjectLoader.from_place('nyc')
    features = FeatureLoader.from_place('')

class SFUniverse(Universe):
    imagery = ImageryLoader.from_place('sf')


config_file



## -> goes to duckdb, that then can be shared with team. Or export duckdb to parquet


Universe(region='nyc', subset:Iterable[nodeids])
# --> NYCUniverse()
    # --> from_subset