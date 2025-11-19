'''Key parts inside the traceback object:

 exc_tb.tb_frame → the current stack frame where the error occurred

exc_tb.tb_frame.f_code → the code object for that frame

exc_tb.tb_frame.f_code.co_filename → the filename of that code

exc_tb.tb_lineno → the line number within that file

error_message: The original error message.
error_detail: The sys module for traceback extraction.

'''


import sys


class Network_Security_Exception(Exception):

    def __init__(self, error_message, error_details):
        self.error_mesaage = error_message
        _,_,exc_tb = error_details.exc_info()


        if exc_tb is not None:
            self.file_name = exc_tb.tb_frame.f_code.co_filename
            self.lineno = exc_tb.tb_lineno

        else:
            self.file_name = "NO TRACEBACK"
            self.lineno = "UNKOWN LINE "

        

    def __str__(self):
        return "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name, self.lineno, str(self.error_mesaage))
        


if __name__ == "__main__":
    
    try:
        a=1/0
    except Exception as e:
        logging.info("exception occured in main")
        raise Network_Security_Exception(e,sys)
    
