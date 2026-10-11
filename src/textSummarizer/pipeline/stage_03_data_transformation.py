from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_transformation import DataTransformation
from textSummarizer.logging import logger
from pathlib import Path

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Check the bouncer's log from the previous stage
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            # Only proceed if the data format was verified
            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.convert()
            else:
                raise Exception("Data Validation Failed. Halting pipeline.")

        except Exception as e:
            print(e)