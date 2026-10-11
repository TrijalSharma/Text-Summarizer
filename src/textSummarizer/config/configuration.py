from textSummarizer.constants import *
from textSummarizer.utils.common import read_yaml, create_directories
from textSummarizer.entity import (DataIngestionConfig, 
                                   DataValidationConfig,
                                   DataTransformationConfig)

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

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        # Automatically create the artifacts/data_validation folder
        create_directories([config.root_dir])

        # Pack the YAML values into the strict entity struct
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        # Create the artifacts/data_transformation folder
        create_directories([config.root_dir])

        # Pack the paths and the tokenizer name into our struct
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name
        )

        return data_transformation_config