
import sys

class Network_Security_Exception(Exception):

    def __init__(self, error_message, error_details):
        self.error_mesaage = error_message
        _,_,exc_tb = error_details.exc_info()

        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.lineno = exc_tb.tb_lineno

    def __str__(self):
        return "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name, self.lineno, str(self.error_mesaage))
        


if __name__ == "__main__":
    try:
        a=1/0
    except Exception as e:
        raise Network_Security_Exception(e,sys)
    
