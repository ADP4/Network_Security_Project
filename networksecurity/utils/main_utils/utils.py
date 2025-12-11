
import yaml

from networksecurity.exception.exception import Network_Security_Exception
from networksecurity.logging.logger import logging

import os, sys
import pandas as pd
import numpy as np
import pickle


def read_yaml_file(file_path:str) -> dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        logging.info("Loading of schema.yaml failed")
        raise Network_Security_Exception(e,sys)
    
def write_yaml_file(file_path, content, replace=False)->None:
    try:
        if replace==True:
            if os.path.exists(file_path):
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content,file)

    except Exception as e:
        logging.info("Writing of yaml file failed in write function")
        raise Network_Security_Exception(e,sys)
    
def read_csv_data(file_path)-> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.info("reading of .csv failed in read function")
        raise Network_Security_Exception(e,sys)

def write_as_csv_data(dataframe:pd.DataFrame,file_path)-> None:
    try:
        return dataframe.to_csv(file_path, index=False,header=True)
    except Exception as e:
        logging.info("wrting of df as csv failed in write function")
        raise Network_Security_Exception(e,sys)
    
def save_numpy_array(filepath:str, array:np.array):
    try:
        os.makedirs(os.path.dirname(filepath),exist_ok=True)
        with open(filepath,"wb") as file_object:
            np.save(file_object,array)
    except Exception as e:
        logging.info("numpy array saving failed")
        raise Network_Security_Exception(e,sys)
    
def load_numpy_array(filepath:str): 
    try:
        with open(filepath,"rb") as file_object:
            return np.load(file_object)
    except Exception as e:
        logging.info("numpy array loading failed")
        raise Network_Security_Exception(e,sys)
    
def save_object (file_path:str, object):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(object, file_obj)
    except Exception as e:
        logging.info("Object saving failed")
        raise Network_Security_Exception(e,sys)
    
def load_object(filepath:str): 
    try:
        if not os.path.exists(filepath):
            raise Exception("{filepath} does not exist")

        with open(filepath,"rb") as file_object:
            print(file_object)
            return pickle.load(file_object)
    except Exception as e:
        logging.info("Object loading failed")
        raise Network_Security_Exception(e,sys)