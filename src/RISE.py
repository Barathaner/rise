
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype

from sklearn import preprocessing
from scipy import stats
import warnings
import numpy as np


class Rule:
    """Class representing a classifier rule, linking a set of conditions with a conclusion.

    Each rule consists of multiple conditions and a final conclusion which is typically used
    in rule-based classification systems.
    """

    def __init__(self, conditions, conclusion):
        # Convert a list of conditions into a dictionary keyed by condition attribute
        # for quicker access and comparisons.
        self._conditions = {condition.get_attribute(): condition for condition in conditions}
        self._conclusion = conclusion

    def __eq__(self, other):
        # Check equality by comparing conditions and conclusion between two rules.
        return self._conditions == other.conditions and self._conclusion == other.conclusion

    def __ne__(self, other):
        # Check inequality by leveraging the equality method.
        return not self == other

    def __hash__(self):
        # Generate a hash based on the conditions and conclusion of the rule.
        # This is needed for rules to be used in sets and as keys in dictionaries.
        return hash((tuple(self._conditions.values()), self._conclusion))

    def __str__(self):
        # Create a human-readable string representation of the rule that lists all conditions
        # followed by the conclusion.
        conditions_str = " && ".join(str(condition) for condition in self._conditions.values())
        return f"{conditions_str} => ({self._conclusion})"

    @property
    def conditions(self):
        # Provide read access to the conditions of the rule.
        return self._conditions

    @property
    def conclusion(self):
        # Provide read access to the rule's conclusion.
        return self._conclusion

    def get_distance_to_instance(self, instance, categorical_class_probability_dict, exponent=2):
        # Calculate the distance between this rule and a given instance. This is used to determine
        # how close an instance is to satisfying the rule, based on categorical and numeric distances.
        return sum(
            self._calculate_distance(attribute, condition, instance, categorical_class_probability_dict, exponent)
            for attribute, condition in self._conditions.items()
        )

    def _calculate_distance(self, attribute, condition, instance, categorical_class_probability_dict, exponent):
        # Determine the distance for a single attribute based on whether the value is categorical
        # or numeric. The specific distance calculation method depends on the type.
        value = instance.loc[attribute]
        dist_type = simplified_value_difference if is_value_categorical(value) else normalized_numeric_range_distance
        dist = dist_type(attribute, value, condition.get_bounds() if dist_type == normalized_numeric_range_distance else condition.get_category(), categorical_class_probability_dict)
        return dist ** exponent

    def is_instance_covered(self, instance):
        # Determine if an instance is covered by the rule, meaning all conditions are met.
        return all(condition.is_respected(instance[attribute]) for attribute, condition in self._conditions.items())

    def get_nearest_non_covered_instance(self, instances, categorical_class_probability_dict):
        # Find the instance closest to the rule that is not currently covered by it.
        # This helps in rule refining and learning processes.
        distances = [
            (instance, self.get_distance_to_instance(instance, categorical_class_probability_dict))
            for instance in instances.itertuples(index=False)
            if not self.is_instance_covered(instance)
        ]
        return min(distances, key=lambda x: x[1], default=(None, np.inf))[0]

    def get_most_specific_generalization(self, instance):
        # Adapt the rule to more broadly cover instances, specifically the one passed that isn't
        # currently covered. This is part of the learning process for refining rules.
        adapted_conditions = [
            self._adapt_condition(attribute, condition, instance[attribute])
            for attribute, condition in self._conditions.items()
        ]
        return Rule(adapted_conditions, self._conclusion)

    def _adapt_condition(self, attribute, condition, instance_value):
        # Modify a condition to cover an instance value if it's not currently covered,
        # particularly by adjusting numeric bounds if necessary.
        if condition.is_respected(instance_value):
            return copy(condition)
        if is_value_numeric(instance_value):
            lower = min(instance_value, condition.get_lower_bound())
            upper = max(instance_value, condition.get_upper_bound())
            return NumericCondition(attribute, lower, upper)


class RuleCoverageAndAccuracy:
    """Class representing a rule, its coverage, and its accuracy."""
    
    def __init__(self, rule, coverage, accuracy):
        self.rule = rule
        self.coverage = coverage
        self.accuracy = accuracy

    def __str__(self):
        return f"{self.rule}\t{{coverage: {round(self.coverage, 3)}, accuracy: {round(self.accuracy, 3)}}}"
class CorrectBoolAndDistance:
    """Class representing a pair of values: a boolean (correct or incorrect) and a distance."""
    
    def __init__(self, is_correct, dist):
        self.is_correct = is_correct
        self.dist = dist

    @property
    def is_correct(self):
        return self._is_correct

    @is_correct.setter
    def is_correct(self, value):
        self._is_correct = value

    @property
    def dist(self):
        return self._dist

    @dist.setter
    def dist(self, value):
        self._dist = value



def pre_process_data_frame(data_frame):

    """Pre-process the passed data frame"""

    # replace the missing values and normalize numeric columns
    return normalize_numeric_columns(replace_missing_values(data_frame))


def fill_nan(series):
    """Fill the NaN values in a series with the mode for categorical data."""
    # Fill NaN with the most frequent value (mode)
    if not is_numeric_dtype(series):
        # Calculate mode using Pandas' mode()
        mode_value = series.mode().dropna()
        if not mode_value.empty:
            return series.fillna(mode_value.iloc[0])
        else:
            return series  # Return original series if mode can't be computed
    else:
        # For numeric data, fill with the mean
        return series.fillna(series.mean())


def replace_missing_values(data_frame):

    """Return the passed data frame with their missing values replaced with the mean of the attribute"""

    # replace NaNs with column means
    data_frame = data_frame.apply(lambda column: fill_nan(column))

    return data_frame

def normalize_numeric_columns(data_frame):

    """ Return the passed data frame with the numeric columns normalized with values between 0 and 1"""

    for column in get_numeric_column_names(data_frame):

        data_frame[column] = preprocessing.minmax_scale(data_frame[column])

    return data_frame

def get_numeric_column_names(data_frame):

    """Return the names of the numeric columns found in the passed data frame"""

    numeric_columns = list()

    for column in data_frame.columns:

        if is_numeric_dtype(data_frame[column]):

            numeric_columns.append(column)

    return numeric_columns




def get_categorical_values(features):
    """returns all names of symbolic features found in the Dataframe of features"""
    
    categorical_columns = list()
    
    for column in features.columns:
        print(column)
        if not is_numeric_dtype(features[column]):
            categorical_columns.append(column)
            
            
    return categorical_columns
    
def get_categorical_probabilities(features,classes):
    """creates a dictionary of probabilities to the corresponding categories. Like Feature Weather has categories sunny, rainy and the classes are yes and no so we have  dictionary {sunny:{yes :P(sunny|yes),no:P(sunny|no)}}"""
    categorical_probabilities_dict = dict()
    categorical_columns = get_categorical_values(features)
    for column in categorical_columns:
        columndict = dict()
        
        for category in features[column].unique():
            categorydict = dict()
            categoryseries = classes[features[column] == category]
            totalcount= categoryseries.count()
             
            for classcategory in classes.T.squeeze().unique():
                
                categorydict[classcategory] = categoryseries[categoryseries == classcategory].count() / totalcount
        
            columndict[category]  = categorydict
    
        categorical_probabilities_dict[column] = columndict
    
    return categorical_probabilities_dict
            
            
def svdm(feature,category0,category1,categoryprobabilitydict,q=1):
    """simplified value difference calculates difference between symbolic values via calculating the absolute differences between their classprobabilities"""
    
    #same categorys have no dist
    if category0==category1:
        return 0

    #get probabilities for categories
    category0dict = categoryprobabilitydict[feature].get(category0,None)
    category1dict = categoryprobabilitydict[feature].get(category1,None)
    
    #return max if missing
    if not category0dict or not category1dict:
        return 1
    
    distance = 0
    
    #apply simplified value difference
    for classcategory in category0dict.keys():
        
        cat0prob = category0dict[classcategory]
        cat1prob = category1dict[classcategory]
        
        distance+= pow(abs(cat0prob - cat1prob),q) 

    return distance


def numeric_distance(value, lower_bound, upper_bound):
    """calculates the distance like RISE is in the papaer, max adn min are the max and min from the attribute in the training set, this was calculated in the preprocessing so we can leave that out"""
    
    # distance is 0 when value is in the bounds
    if lower_bound <= value <= upper_bound:
        return 0

    if value > upper_bound:
        return (value-upper_bound)
    if value < lower_bound:
        return(lower_bound-value)


def create_rule_from_instance(instance, class_field, class_value):

    """Create and return a rule based on the passed instance"""

    conditions = list()

    # for each field, make a condition equal to the instance value
    for field, value in instance.to_dict().items():
        
        # categorical fields have equality-based conditions
        if not is_numeric_dtype(value):
            conditions.append(CategoricalCondition(field, value))
            
        # numeric fields have conditions expressed as a degenerate range (lower and upper bounds are equal)
        else:
            conditions.append(NumericCondition(field, value, value))
            
    # add the class field-value match as conclusion
    conclusion = Conclusion(class_field, class_value)

    return Rule(conditions, conclusion)


def get_rule_set_accuracy(rule_set, instances, class_values, class_field, categorical_class_probability_dict, use_leave_one_out, rule_must_cover_instance=False):

    """Calculate and return the accuracy of the passed rule set for the instances defined by the feature matrix x and the class vector y"""

    correct_classification_num = 0

    assigned_classes = list()

    # dictionary that will associate each instance index with the classification result (correct or not) and the distance to its nearest rule
    instance_result_dict = dict()

    # classify each instance using its nearest rule, leaving out the instance-based rule
    for index, instance in instances.iterrows():

        instance_rule = create_rule_from_instance(instance, class_field, class_values[index])

        # ignore the instance's own rule if necessary
        rule_to_ignore = None
        if use_leave_one_out:
            rule_to_ignore = instance_rule

        nearest_rule, nearest_dist = get_nearest_rule(rule_set, instance, categorical_class_probability_dict, rule_to_ignore, rule_must_cover_instance)

        if nearest_rule:

            assigned_class = nearest_rule.get_conclusion().get_value()
            assigned_classes.append(assigned_class)

            is_correct = assigned_class == class_values[index]
            if is_correct:
                correct_classification_num += 1

        else:
            is_correct = False

        instance_result_dict[index] = CorrectBoolAndDistance(is_correct, nearest_dist)

    return correct_classification_num / len(class_values), assigned_classes, instance_result_dict

def get_nearest_rule(rule_set, instance, categorical_class_probability_dict, rule_to_ignore=None, rule_must_cover_instance=False):

    """Return the rule from the passed rule set that is more similar to the passed instance"""

    min_dist = np.inf
    nearest_rule = None

    # find the rule with minimum distance to the instance
    for rule in rule_set:

        if not rule_to_ignore or rule != rule_to_ignore:

            if not rule_must_cover_instance or rule.is_instance_covered(instance):

                dist = rule.get_distance_to_instance(instance, categorical_class_probability_dict)

                if dist < min_dist:
                    min_dist = dist
                    nearest_rule = rule

    return nearest_rule, min_dist

def do_rise_algorithm(featurematrix,classvector):
    """Build a RISE rule-based/instance-based classifier using the passed feature matrix and the class vector, and return the classifier's rules"""

    # get the name of the class field
    class_field = classvector.name
    # let ruleset be the initially featurematrix
    rule_set = {create_rule_from_instance(featurematrix, class_field, classvector[i]) for i, feature in featurematrix.iterrows()}
    # for the sake of faster computation, only calculate once the probabilities of each category (of all the categorical values of x) to produce each class of y
    categorical_class_probability_dict = get_categorical_probabilities(featurematrix, classvector)
    
    # compute the initial accuracy, and get the dictionary of results and distances that will allow later incremental updates of the accuracy
    initial_accuracy, _, featurematrix_result_dict = get_rule_set_accuracy(rule_set, featurematrix, classvector, class_field, categorical_class_probability_dict, True)
    accuracy = initial_accuracy
    print("\t\tInitial accuracy:", round(accuracy, 3))

    has_accuracy_increased = True

    
    return 0