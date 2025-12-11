
from networksecurity.exception.exception import Network_Security_Exception
from networksecurity.logging.logger import logging

import os, sys
import pandas as pd
import numpy as np

from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from networksecurity.utils.main_utils.utils import read_csv_data,save_numpy_array, save_object
from networksecurity.constant.training_pipeline_constants import TARGET_COLUMN

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            logging.info("data transformation object initialised")

        except Exception as e:
            logging.info("data transformation object initialisation failed")
            raise Network_Security_Exception(e,sys)

    def get_preprocessor_object (self)->Pipeline:
        try:
            logging.info("initalising preprocessor object in data transformation class")
            scalar = StandardScaler()
            preprocessor = Pipeline(steps=[("scalar", scalar)])
            return preprocessor
        except Exception as e:
            logging.info("failed: initalising preprocessor object in data transformation class")
            raise Network_Security_Exception(e,sys)
        
    def initiate_data_transformation(self)-> DataTransformationArtifact:
        try:
            train_filepath = self.data_validation_artifact.valid_train_file_path
            test_filepath= self.data_validation_artifact.valid_test_file_path

            logging.info(f"Reading validated train data")
            train_df= read_csv_data(train_filepath)
            logging.info(f"Reading validated test data")
            test_df= read_csv_data(test_filepath)

            X_train = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            y_train = train_df[TARGET_COLUMN]
            y_train = y_train.replace(-1,0)

            X_test = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            y_test = test_df[TARGET_COLUMN]
            y_test = y_test.replace(-1,0)

            preprocessor = self.get_preprocessor_object()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_transformed,np.array(y_train)]
            test_arr =  np.c_[X_test_transformed, np.array(y_test)]

            save_numpy_array(self.data_transformation_config.transformed_data_train_file_path,train_arr)
            save_numpy_array(self.data_transformation_config.transformed_data_test_file_path,test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_data_test_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_data_train_file_path)
            
            return data_transformation_artifact

        except Exception as e:
            logging.info("generation of data transformation artifact failed")
            raise Network_Security_Exception(e,sys)