"""
This file has some functions to capture the exceptions and save them in a file.

@author: Masoumeh Moradipour Tari
@date: September 2019
@place: Germany, University of Tuebingen
"""


def save_as_exception(root, file, message):
    with open("./Result/exceptions.txt", 'a+') as txt:
        txt.write('\r\n')
        txt.write("###################################################################################"+'\r\n')
        txt.write("##################################### Exception ###################################"+'\r\n')
        txt.write("###################################################################################"+'\r\n')
        txt.write('\r\n')
        txt.write("Directory: " + root + '\r\n')
        txt.write("File: " + file + '\r\n')
        txt.write("Error Message: " + str(message) + '\r\n')
