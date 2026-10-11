from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from textSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from textSummarizer.logging import logger

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME_2 = "Data Validation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME_2} started <<<<<<") 
   data_validation = DataValidationTrainingPipeline()
   data_validation.run_validation()
   logger.info(f">>>>>> stage {STAGE_NAME_2} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME_3 = "Data Transformation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME_3} started <<<<<<") 
   data_transformation = DataTransformationTrainingPipeline()
   data_transformation.main()
   logger.info(f">>>>>> stage {STAGE_NAME_3} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e