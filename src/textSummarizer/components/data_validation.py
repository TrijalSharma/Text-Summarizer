import os
from textSummarizer.logging import logger
from textSummarizer.entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            validation_status = True
            
            # 1. Look inside the extracted dataset folder
            dataset_path = os.path.join("artifacts", "data_ingestion", "samsum_dataset")
            all_files = os.listdir(dataset_path)
            
            # 2. Check if every required file exists in that folder
            for req_file in self.config.ALL_REQUIRED_FILES:
                if req_file not in all_files:
                    validation_status = False
                    break
                    
            # 3. Write the final verdict to the status.txt file
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            
            # 4. Log the result to the terminal
            if validation_status:
                logger.info("All required files found! Data Validation Successful.")
            else:
                logger.error("Missing required files! Data Validation Failed.")
                
            return validation_status
        
        except Exception as e:
            raise e