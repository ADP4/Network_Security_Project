
from networksecurity.exception.exception import Network_Security_Exception
from networksecurity.logging.logger import logging

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig

import sys

if __name__=='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()

        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)

        logging.info("main: Initiate the data ingestion")
        dataingestionartifact=data_ingestion.data_ingestion_initiation()
        logging.info("main: Data Initiation Completed")
        print("================================================================================")
        print("Ingestion artifact:")
        print(dataingestionartifact)

        datavalidationconfig=DataValidationConfig(trainingpipelineconfig)
        datavalidationartifact=DataValidation(datavalidationconfig,dataingestionartifact)
        logging.info("main: Initiate the data validation")
        dataingestionartifact=data_ingestion.data_ingestion_initiation()
        logging.info("main: Data Validation Completed")
        print("================================================================================")
        print("Validation artifact:")
        print(datavalidationartifact)

    except Exception as e:
        logging.info("Error in main")
        raise Network_Security_Exception(e,sys)