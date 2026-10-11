from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_validation import DataValidation
from textSummarizer.logging import logger

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def run_validation(self):
        # Initialize manager and get the strict validation paths
        config_manager = ConfigurationManager()
        validation_config = config_manager.get_data_validation_config()
        
        # Pass the paths to our "bouncer" component and pull the trigger
        validator = DataValidation(config=validation_config)
        validator.validate_all_files_exist()