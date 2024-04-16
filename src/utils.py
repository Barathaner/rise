
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype
import numbers
from sklearn import preprocessing
from scipy import stats
import warnings
import os

def is_value_numeric(value):

    """Return if the passed value is numeric"""

    return isinstance(value, numbers.Number)


def is_value_categorical(value):

    """Return if a value is categorical"""

    return type(value) == str

def round_dict_values(dictionary, decimals):

    """Round all the numeric values in the passed dictionary"""

    for key, value in dictionary.items():

        if is_value_numeric(value):

            dictionary[key] = round(value, decimals)

    return dictionary

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
def split_features_and_class(data_frame, data_set_name):
    """Split the DataFrame into features (X) and class (y) based on the dataset's structure."""
    if data_set_name == "agaricus-lepiota":
        # For 'mushrooms' where the class column is the first
        y = data_frame.iloc[:, 0]
        X = data_frame.iloc[:, 1:]
    else:
        # For other datasets where the class column is the last
        y = data_frame.iloc[:, -1]
        X = data_frame.iloc[:, :-1]
    return X, y
