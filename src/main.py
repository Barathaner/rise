
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
    data_set_names = ["Recommended_Lenses", "transfusion", "car"]
    
    dataframes = read_data_sets(data_dir_path,data_set_names)
    # pre-process the data sets
    data_frames= [pre_process_data_frame(data_frame) for data_frame in dataframes]
    
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
        rule_set_with_coverage_and_accuracy, categorical_class_probability_dict = do_rise_algorithm(x_train, y_train)

            # show the rules
        print_and_write("\t\nRule set:".format(len(rule_set_with_coverage_and_accuracy)), data_set_file_name)
        for i, rule_with_coverage_and_accuracy in enumerate(rule_set_with_coverage_and_accuracy):
            print_and_write("\t\t\t({}) {}".format(i+1, rule_with_coverage_and_accuracy), data_set_file_name)

        rule_set = {rule_with_coverage_and_accuracy.get_rule() for rule_with_coverage_and_accuracy in rule_set_with_coverage_and_accuracy}

        # evaluate the rules on the test set to assess the classifier's accuracy
        accuracy, assigned_classes = evaluate_rule_set(rule_set, x_test, y_test, categorical_class_probability_dict)
        accuracies.append(accuracy)

        # show the assigned classes for each test instance
        print_and_write("\t\tTest instances with assigned classes:", data_set_file_name)
        for i, ((_, instance), real_class, assigned_class) in enumerate(zip(x_test.iterrows(), y_test, assigned_classes)):
            print_and_write("\t\t\t({}) Attributes: {},\nReal class: {},\nPredicted class: {}".format(i+1, round_dict_values(instance.to_dict(), 3), real_class, assigned_class), data_set_file_name)

        print_and_write("\t\tTest accuracy: {}".format(round(accuracy, 3)), data_set_file_name)

        
