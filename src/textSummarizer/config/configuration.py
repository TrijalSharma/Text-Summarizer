from textSummarizer.constants import *
from textSummarizer.utils.common import read_yaml, create_directories
from textSummarizer.entity import DataIngestionConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        # 1. Read the YAML files
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # 2. Automatically create the master "artifacts" folder
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        # 3. Automatically create the "artifacts/data_ingestion" subfolder
        create_directories([config.root_dir])

        # 4. Lock the paths into our custom C++ style "struct" (the dataclass)
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config