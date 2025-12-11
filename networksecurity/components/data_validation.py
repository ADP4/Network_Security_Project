
import os
import sys

from networksecurity.exception.exception import Network_Security_Exception
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

from networksecurity.constant.training_pipeline_constants import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file, read_csv_data, write_as_csv_data


from scipy.stats import ks_2samp
import pandas as pd


class DataValidation:
    def __init__(self, data_validaion_config:DataValidationConfig, 
                 data_ingestion_artifact:DataIngestionArtifact):
        try:

            self.data_validation_config = data_validaion_config
            self.data_ingestion_artifact = data_ingestion_artifact

            # Load schema configuration from YAML
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logging.info(f"Loaded schema config from: {SCHEMA_FILE_PATH}")

        except Exception as e:
            raise Network_Security_Exception(e,sys)
        

    def validate_number_of_columns(self, dataframe:pd.DataFrame) -> bool:

        """
        Validates that the number of columns in the dataframe matches
        the expected number of columns defined in the schema.

        if validated the return true
        """

        try:

            required_num_of_columns = len(self.schema_config.get("columns",{}))
            num_of_columns_in_df= len(dataframe.columns)

            logging.info(f"required number of columns are: {required_num_of_columns}")
            logging.info(f"number of columns in dataframe are: {num_of_columns_in_df}")

            return required_num_of_columns==num_of_columns_in_df
        
        except Exception as e:
            raise Network_Security_Exception (e,sys)
        
    def validate_column_names(self, dataframe:pd.DataFrame)->bool:

        """
        Validates that all expected columns from schema are present in the dataframe.
        Also checks for unexpected extra columns.

        if validated returns true
        """
        logging.info("column name validation started")
        try:
            #test for duplicate columns in dataframe
            if len(list(dataframe.columns)) != len(list(set(dataframe.columns))):
                logging.info("duplicate columns in dataframe detected")
                return False
            logging.info("duplicate columns in dataframe NOT detected")

            #test of missing or extra columns
            actual_columns = set(dataframe.columns)
            excepted_columns = set(self.schema_config.get("columns",{}))

            missing_columns = excepted_columns - actual_columns
            if missing_columns:
                logging.info(f"missing columns in dataframe: {missing_columns}")
            logging.info(f" NO missing columns in dataframe {missing_columns}")


            extra_columns = actual_columns - excepted_columns
            if extra_columns:
                logging.info(f" Extra columns in dataframe: {extra_columns}")
            logging.info(f" NO Extra columns in dataframe {extra_columns}")

            if len(missing_columns)!=0 or len(extra_columns)!=0:
                logging.info("------in if false")
                return False
            else:
                logging.info("------true")
                return True
            
                
        except Exception as e:
            raise Network_Security_Exception (e,sys)
        
    def detect_dataset_drift(self, base_df, current_df, threshold = 0.05)->bool:
        '''
        detects drift and writes drift report
        base_df:trainig dataset
        current_df: test dataset
        drift detected then returns True
        '''
        try:
            #status turns true if drift is found
            status = False
            report={}

            for column in base_df.columns:
                base_col = base_df[column]
                current_col = current_df[column]
                ks_result = ks_2samp(base_col,current_col)
                pvalue = float(ks_result.pvalue)

                # If p_value < threshold => distributions are significantly different => drift
                drfit_found = pvalue < threshold

                if drfit_found:
                    status=True
                report[column]= {"pvalue":pvalue,
                                "drift_status":drfit_found}
                
            #writing yaml file
            report_file_path = self.data_validation_config.drift_report_file_path

            logging.info(f"Writing drift report to: {report_file_path}")
            write_yaml_file(file_path=report_file_path,content=report)
            logging.info("Writing drift report successful")

            return status
        
        except Exception as e:
            logging.info("Drift report generation failed")
            raise Network_Security_Exception (e,sys)
        
    def initiate_data_validation(self):

        try:
            #loading test train df
            train_path = self.data_ingestion_artifact.trained_file_path
            test_path= self.data_ingestion_artifact.test_file_path
            train_df= read_csv_data(train_path)
            test_df=read_csv_data(test_path)

            errors=[]

            #validate number of columns
            if not self.validate_number_of_columns(train_df):
                errors.append("Train df does not contain all required columns")
            if not self.validate_number_of_columns(test_df):
                errors.append("Test df does not contain all required columns")

            #validate column names
            if not self.validate_column_names(train_df):
                errors.append("Train df columns do not match the schema")
            if not self.validate_column_names(test_df):
                errors.append("Test df columns do not match the schema")

            if errors:
                all_errors= "\n".join(errors)
                print (all_errors)
                logging.info(f"Data Validation has failed:\n{all_errors}")
                raise Network_Security_Exception("validation failed: all errors",sys)
            
            #check for the drift in data
            drift_status = self.detect_dataset_drift(train_df,test_df,threshold=0.05)

            #writing validated data
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_as_csv_data(train_df,self.data_validation_config.valid_train_file_path)
            write_as_csv_data(test_df,self.data_validation_config.valid_test_file_path)

            logging.info("valid train and test written as csv")

            data_validation_artifact= DataValidationArtifact(validation_status=drift_status,
                                                             valid_test_file_path=self.data_validation_config.valid_test_file_path,
                                                             valid_train_file_path=self.data_validation_config.valid_train_file_path,
                                                             invalid_test_file_path=None, invalid_train_file_path=None,
                                                             drift_report_file_path=self.data_validation_config.drift_report_file_path)
            logging.info("data validation artifaction created and data validation completed")
            return data_validation_artifact
        except Exception as e:
            logging.info("data validation orchestrater failed")
            raise Network_Security_Exception(e,sys)
        


        









    

    
        
    
