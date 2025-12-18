
from networksecurity.exception.exception import Network_Security_Exception
from networksecurity.logging.logger import logging

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig

import sys

if __name__=='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()

        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)

        logging.info("main: Initiate the data ingestion")
        dataingestion_artifact=data_ingestion.data_ingestion_initiation()
        logging.info("main: Data Initiation Completed")
        print("================================================================================")
        print("Ingestion artifact:")
        print(dataingestion_artifact)

        datavalidationconfig=DataValidationConfig(trainingpipelineconfig)
        data_validation=DataValidation(datavalidationconfig,dataingestion_artifact)

        logging.info("main: Initiate the data validation")
        datavalidation_artifact=data_validation.initiate_data_validation()
        logging.info("main: Data Validation Completed")
        print("================================================================================")
        print("Validation artifact:")
        print(datavalidation_artifact)

        datatransformationconfig = DataTransformationConfig(trainingpipelineconfig)
        data_transformation = DataTransformation(datavalidation_artifact,datatransformationconfig)

        logging.info("main: Initiate the data transformation")
        datatransformation_artifact=data_transformation.initiate_data_transformation()
        logging.info("main: Data Transformation Completed")
        print("================================================================================")
        print("transformation artifact:")
        print(datatransformation_artifact)

        modeltrainerconfig = ModelTrainerConfig(trainingpipelineconfig)
        model_trainer = ModelTrainer(datatransformation_artifact, modeltrainerconfig)

        logging.info("main: Initiate the model training")
        modeltrainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("main: Model Training Completed")
        print("================================================================================")
        print("Model Trainer Artifact:")
        print(modeltrainer_artifact)


    except Exception as e:
        logging.info("Error in main")
        raise Network_Security_Exception(e,sys)