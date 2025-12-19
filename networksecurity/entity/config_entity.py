
from datetime import datetime
import os

from networksecurity.constant import training_pipeline_constants

class TrainingPipelineConfig:
    '''  Initial configuration '''

    def __init__(self, timestamp: str| None = None):

        if timestamp is None:
            timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        self.timestamp=timestamp

        self.pipeline_name = training_pipeline_constants.PIPELINE_NAME
        self.artifact_dir_name = training_pipeline_constants.ARTIFACT_DIR_NAME

        self.artifact_timestamp_dir_path = os.path.join(self.artifact_dir_name,self.timestamp)

        self.model_dir_name = os.path.join("final_model")

        

class DataIngestionConfig:
    ''' Configuration for ingesting data '''

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir_path = os.path.join(training_pipeline_config.artifact_timestamp_dir_path,
                                                    training_pipeline_constants.DATA_INGESTION_DIR_NAME)
        
        self.feature_store_file_path = os.path.join(self.data_ingestion_dir_path,
                                                    training_pipeline_constants.DATA_INGESTION_FEATURE_STORE_DIR_NAME,
                                                    training_pipeline_constants.FEATURE_STORE_FILE_NAME)
        
        self.train_data_file_path = os.path.join(self.data_ingestion_dir_path,
                                                 training_pipeline_constants.DATA_INGESTION_INGESTED_DIR_NAME,
                                                 training_pipeline_constants.TRAIN_FILE_NAME)
        
        self.test_data_file_path = os.path.join(self.data_ingestion_dir_path,
                                                 training_pipeline_constants.DATA_INGESTION_INGESTED_DIR_NAME,
                                                 training_pipeline_constants.TEST_FILE_NAME)
        
        self.database_name = training_pipeline_constants.DATA_INGESTION_DATABASE_NAME
        self.collection_name = training_pipeline_constants.DATA_INGESTION_COLLECTION_NAME
        self.train_test_split_ratio = training_pipeline_constants.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

class DataValidationConfig:
        
    '''Configuration for validation of data'''

    def __init__(self,train_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir_path = os.path.join(train_pipeline_config.artifact_timestamp_dir_path,
                                                         training_pipeline_constants.DATA_VALIDATION_DIR_NAME)
            
        self.valid_train_file_path = os.path.join(self.data_validation_dir_path,
                                                      training_pipeline_constants.DATA_VALIDATION_VALID_DIR_NAME,
                                                      training_pipeline_constants.TRAIN_FILE_NAME)
            
        self.valid_test_file_path =os.path.join(self.data_validation_dir_path,
                                                    training_pipeline_constants.DATA_VALIDATION_VALID_DIR_NAME,
                                                    training_pipeline_constants.TEST_FILE_NAME)
            
        self.invalid_train_file_path = os.path.join(self.data_validation_dir_path,
                                                        training_pipeline_constants.DATA_VALIDATION_INVALID_DIR_NAME,
                                                        training_pipeline_constants.TRAIN_FILE_NAME)
            
        self.invalid_test_file_path =os.path.join(self.data_validation_dir_path,
                                                    training_pipeline_constants.DATA_VALIDATION_INVALID_DIR_NAME,
                                                    training_pipeline_constants.TEST_FILE_NAME)
            
        self.drift_report_file_path = os.path.join(self.data_validation_dir_path,
                                                       training_pipeline_constants.DATA_VALIDATION_DRIFT_REPORT_DIR_NAME,
                                                       training_pipeline_constants.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
        
class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir_path = os.path.join(training_pipeline_config.artifact_timestamp_dir_path,
                                                         training_pipeline_constants.DATA_TRANSFORMATION_DIR_NAME)
        
        self.transformed_data_train_file_path = os.path.join(self.data_transformation_dir_path,
                                                             training_pipeline_constants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR_NAME,
                                                             training_pipeline_constants.TRAIN_FILE_NAME.replace(".csv",".npy"))
        
        self.transformed_data_test_file_path = os.path.join(self.data_transformation_dir_path,
                                                             training_pipeline_constants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR_NAME,
                                                             training_pipeline_constants.TEST_FILE_NAME.replace(".csv",".npy"))
        
        self.transformed_object_file_path = os.path.join(self.data_transformation_dir_path,
                                                         training_pipeline_constants.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR_NAME,
                                                         training_pipeline_constants.PREPROCESSING_OBJECT_FILE_NAME)
        
class ModelTrainerConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir_path = os.path.join(training_pipeline_config.artifact_timestamp_dir_path,
                                                   training_pipeline_constants.MODEL_TRAINER_DIR_NAME)
        
        self.trained_model_file_path = os.path.join(self.model_trainer_dir_path,
                                                    training_pipeline_constants.MODEL_TRAINER_TRAINED_MODEL_DIR_NAME,
                                                    training_pipeline_constants.MODEL_TRAINER_TRAINED_MODEL_NAME)
        
        self.expected_accuracy = training_pipeline_constants.MODEL_TRAINER_EXPECTED_SCORE
        self.underfitting_overfitting_threshold = training_pipeline_constants.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD
        

        
            

    

        

