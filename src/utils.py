
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype

from sklearn import preprocessing
from scipy import stats
import warnings
import os

def clear_file(file_name):
    """Clear the contents of a file by ensuring the directory exists first."""

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # Clear any previous content in the file
    with open(file_name, "w") as file:
        file.close()

def print_and_write(message, file_name):

    """Print the passed string in the console and write it to the current global file"""

    print(message)
    file = open(file_name, "a")
    file.write(message + "\n")
    file.close()
    
def read_data_sets(data_dir_path, data_set_names):

    """Read and return as data frames the data sets under the passed directory path that have the passed names"""

    data_frames = list()

    # parse the data sets
    for data_set_name in data_set_names:

        data_frame = pd.read_csv(data_dir_path + data_set_name + ".csv", na_values=["?"])
        data_frame.name = data_set_name
        data_frames.append(data_frame)


    return data_frames

def split_features_and_class(dataframe,datasetname):
    """splits dataset in featurematrix and classvector"""
    class_column_name = dataframe.columns[-1]
    
    if (datasetname=='agaricus-lepiota'):
        class_column_name=dataframe.columns[0]
        
    y = dataframe.pop(class_column_name).to_frame()
    x = dataframe
    return x,y