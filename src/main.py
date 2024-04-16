
from utils import *
from RISE import *
from pathlib import Path

if __name__ == "__main__":
    # path to the data sets
    parent_dir_path = str(Path(__file__).parents[1])
    data_dir_path = parent_dir_path + "/Data/"
    results_dir_path = parent_dir_path + "/Results/"
    # data sets to use
    #lenses this has 28 instances
    #transfusion this has 748 instances
    #mushrooms this has 8124 instances
    data_set_names = ["Recommended_Lenses", "transfusion", "agaricus-lepiota"]
    
    dataframes = read_data_sets(data_dir_path,data_set_names)
    # pre-process the data sets
    data_frames = [pre_process_data_frame(data_frame) for data_frame in dataframes]
    
    for data_set_name, data_frame in zip(data_set_names, data_frames):
       # prepare the file where to write the results
        data_set_file_name = results_dir_path + data_set_name + ".txt"
        clear_file(data_set_file_name)
        print_and_write("Data set: {}".format(data_set_name), data_set_file_name)

        #split dataset in test and train dataset
        train,test = train_test_split(data_frame,test_size=0.2) 
        
        # seperate handling, because class column for mushrooms is the first column, at the other datasets it is the last column
        x_train,y_train= split_features_and_class(train,data_set_name)
        x_test,y_test= split_features_and_class(test,data_set_name)
        accuracies = list()
        print_and_write("\t Training model ...", data_set_file_name)


        # use the training set to build the classifier and produce the model's rules
        rule_set_with_coverage_and_accuracy, categorical_class_probability_dict = do_rise_algorithm(x_train, y_train, True)

        
