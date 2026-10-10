from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_ingestion import DataIngestion
from textSummarizer.logging import logger

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # 1. Initialize the manager to read the YAML files
        config = ConfigurationManager()
        
        # 2. Get the specific data ingestion paths packed in our struct
        data_ingestion_config = config.get_data_ingestion_config()
        
        # 3. Pass those paths to the component engine
        data_ingestion = DataIngestion(config=data_ingestion_config)
        
        # 4. Pull the trigger to download and extract
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()