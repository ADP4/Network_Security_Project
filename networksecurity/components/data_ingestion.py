from networksecurity.exception.exception import Network_Security_Exception
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

import pandas as pd
import numpy as np
import pymongo 
import sys
import os
from sklearn.model_selection import train_test_split

import certifi
CA_file = certifi.where()

from dotenv import load_dotenv
load_dotenv()

mongodb_url = os.getenv("MONGO_DB_URL")

if mongodb_url is None:
    logging.info("Mongo db url is empty")
    raise Network_Security_Exception("Empty MongoDb URL", sys)

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            
            self.data_ingestion_config = data_ingestion_config
            logging.info(f"DataIngestion initialized with config: {self.data_ingestion_config}")

        except Exception as e:
            logging.info("data ingestion config failed")
            raise Network_Security_Exception(e, sys)

    def export_collection_as_dataframe(self):
        
        try:

            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(mongodb_url, tls=True, tlsCAFile = CA_file)
            
            logging.info("Mongo client created and now reading from database")
            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            logging.info("Dataframe generated")


            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis = 1)

            df.replace({"na":np.nan},inplace=True)

            return df
        
        except Exception as e:
            logging.info("exporting collection as dataframe failed")
            raise Network_Security_Exception(e,sys)
    
    def export_data_to_feature_store(self, dataframe:pd.DataFrame):
        try:
            feature_store_file_path = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_file_path,exist_ok=True)
            logging.info("feature store dir created")
            dataframe.to_csv(self.data_ingestion_config.feature_store_file_path,
                             index=False, header=True)
            logging.info("Data export to feature store completed successfully.")

        except Exception as e:
            logging.info("export to feature store failed")
            raise Network_Security_Exception(e,sys)
        
    def split_data_as_train_test(self, dataframe:pd.DataFrame):
        
        try:

            train_df, test_df = train_test_split(dataframe, test_size =self.data_ingestion_config.train_test_split_ratio)
            logging.info("train test split successful")

            dir_path = os.path.dirname(self.data_ingestion_config.train_data_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info("Directory to store train test files created")

            train_df.to_csv(self.data_ingestion_config.train_data_file_path, index=False, header=True)
            test_df.to_csv(self.data_ingestion_config.test_data_file_path, index=False, header=True)

            logging.info("train and test datafiles exported successfully")

        except Exception as e:
            logging.info("train test exportation failed")
            raise Network_Security_Exception(e,sys)
        
    def data_ingestion_initiation(self):
        try:

            dataframe = self.export_collection_as_dataframe()
            self.export_data_to_feature_store(dataframe=dataframe)
            self.split_data_as_train_test(dataframe=dataframe)
            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.train_data_file_path,
                                                            test_file_path=self.data_ingestion_config.test_data_file_path)
            return data_ingestion_artifact
            logging.info("data ingestion successful")
        except Exception as e:
            logging.info("data ingestion initiation failed")
            raise Network_Security_Exception(e,sys)
    
        


