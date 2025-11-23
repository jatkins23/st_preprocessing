from pathlib import Path
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    ddb_path: Path = Path(str(os.getenv('DDB_PATH')))
    data_path: Path = Path(str(os.getenv('DATA_PATH')))
    proj_crs: str = str(os.getenv('PROJ_CRS'))
    export_path: Path = Path(str(os.getenv('EXPORT_PATH')))

    class Config: 
        env_prefix = ""
        env_file   = ".env"

settings = Settings()
print(settings)