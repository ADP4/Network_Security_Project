"""
Module: push_data.py
Purpose:
    - Read phishing dataset from CSV
    - Convert it into JSON-like Python dictionaries
    - Insert the data into MongoDB Atlas

This is part of the ETL (Extract–Transform–Load) flow.
"""
import sys
import os
from dotenv import load_dotenv
import pandas as pd
import pymongo

import certifi
CA_file = certifi.where()

from networksecurity.exception.exception import Network_Security_Exception
from networksecurity.logging.logger import logging

#reading mongo db url from .env
load_dotenv()
mongodb_url = os.getenv("MONGO_DB_URL")

if mongodb_url is None:
    logging.info("Mongo db url is empty")
    raise Network_Security_Exception("Empty MongoDb URL", sys)


class Network_Security_ExtractData:

    '''handles csv to json like convesrion and data insertion to mongoDb'''

    #Initialsing constructor for the class 

    def __init__(self):
        try:
            pass
        except Exception as e:
            logging.info("Error in initialisation of constructor for Extract Data")
            raise Network_Security_Exception(e, sys)
        
    def convert_to_json_like (self, file_path):
        '''
        Reading data -> converting to json like

        '''

        try:
            data = pd.read_csv(file_path)
            logging.info("csv file loaded successfully")


            # reseted the index and droped original index
            data.reset_index(drop=True, inplace=True)

            #conversion to json like
            record = data.to_dict(orient="records")
            logging.info("conversion to json like successful")

        except Exception as e:
            logging.info("error in convert_to_json_like")
            raise Network_Security_Exception(e, sys)

        return record
    
    def data_insertion_MongoDb (self, record, database, collection):

        try:

            self.database = database
            self.record = record
            self.collection = collection
            
            self.mongo_client= pymongo.MongoClient(mongodb_url, tls = True, tlsCAFile = CA_file) 
            logging.info("Mongo client created")

            #selecting database and collection
            database1= self.mongo_client[self.database]
            logging.info("db name initialised")
            collection1 = database1[self.collection]
            logging.info("coll name initialised")

            logging.info(f'Inserting {len(self.record)} in database: {self.database} collection: {self.collection}')

            inserted_record = collection1.insert_many(self.record)

            logging.info(f'INSERTED {len(inserted_record.inserted_ids)} in database: {database} collection: {collection}')

            return inserted_record.inserted_ids


        except Exception as e:
            logging.info("insertion of records to mongodb failed")
            raise Network_Security_Exception(e,sys)
        
if __name__ == "__main__":

    try:

        logging.info("ETL started")
        
        file_path=os.path.join("Network_Data","phisingData.csv")
        database_name="MLProjects"
        collection_name="NetworkData"

        netwrokobj = Network_Security_ExtractData()

        records = netwrokobj.convert_to_json_like(file_path=file_path)

        len_record = netwrokobj.data_insertion_MongoDb(
            record=records, 
            database=database_name, 
            collection=collection_name)

        logging.info("ETL Completed: with insertion of {len_record} records")


    except Exception as e:
        logging.info("Error occurred in main while executing ETL")
        raise Network_Security_Exception(e,sys)


    
