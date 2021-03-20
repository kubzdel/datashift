from abc import abstractmethod, ABC
from enum import Enum
from typing import Union


class TaskType(Enum):
    FILTER = 1
    PROCESSOR = 2
    BALANCING = 3
    REDUCER = 4


class _AbstractTask(ABC):
    """
    Generic abstract class representing a single task to be performed on a single data observation.
    """

    @abstractmethod
    def type(self) -> TaskType:
        """
        Returns the type of the given task. This method must be overridden by an appropriate method in the specialized class.
        Returns:
            TaskType: Enumerator characterizing the type of a given task.
        """
        pass

    def get_chunk_size(self):
        return 1

    def setup(self):
        pass

    def teardown(self):
        pass

    def assign_proxy_object(self,proxy_object):
        pass

class AbstractFilterTask(_AbstractTask):
    """
    Generic abstract class representing task filtering out the observations that do not meet the assumed criteria.
    This class must be inherited by a specialized class.
    """

    def type(self) -> TaskType:
        """
        Returns the type of the given task - filtering task.

        Returns:
            TaskType: Enumerator characterizing the type of a given task.
        """
        return TaskType.FILTER

    @abstractmethod
    def filter(self, sample) -> bool:
        """
        Calculates whether an observation should be passed on to the next pipeline step or be filtered out at a given stage.
        This class must be inherited by a specialized class. This method must be overridden by an appropriate method in the specialized class.

        Args:
            sample (dict): The single data observation in which the elements of the dictionary are individual features (e.g. values in each column of the csv file).

        Returns:
            bool: True for passing on the observation to the next step, False for rejecting the observation processing at this stage.
        """
        pass

class NotNoneFilterTask(AbstractFilterTask):

    def filter(self, sample) -> bool:
        return sample is not None


class AbstractProcessingTask(_AbstractTask):
    """
      Generic abstract class representing task processing a given data sample.
      This task can process and change values in the dictionary (modifying, deleting, adding new elements/fields). Changes made to the data observation are passed on to the next step.
      This class must be inherited by a specialized class.
    """

    def type(self) -> TaskType:
        """
        Returns the type of the given task - processing task.

        Returns:
            TaskType: Enumerator characterizing the type of a given task.
        """
        return TaskType.PROCESSOR

    @abstractmethod
    def process(self, sample) -> dict:
        """
        Processes and modifies data observation.

        Args:
            sample (dict): The single data observation in which the elements of the dictionary are individual features (e.g. values in each column of the csv file).

        Returns:
            dict: Modified data observation
        """
        pass

class AbstractReduceTask(_AbstractTask):
    """
      Generic abstract class representing task turning observations into numerical values (int, float) or observations into a dictionary of numerical values (dict, int, float).
      This task will be useful in particular when generating a summary of the final data set.
      Thanks to the use of reducing tasks after processing, filtering and balancing tasks, it is possible to generate metadata of a data set (e.g. mean values, max text lenth, numer of samples per category) without re-iteration over a whole data.
      This class must be inherited by a specialized class.

    Args:
        reduced_value_name (str): Name of the parameter specifying a reduced value (int, float, dict). This name will be used as the key of the given value in the generated json or yaml file with reduced values.

    Attributes:
        reduced_value_name (str): Name of the parameter specifying a reduced value (int, float, dict). This name will be used as the key of the given value in the generated json or yaml file with reduced values.
    """

    def __init__(self, reduced_value_name=None):
        self.reduced_value_name = reduced_value_name

    def type(self) -> TaskType:
        """
        Returns the type of the given task - reducing task.

        Returns:
            TaskType: Enumerator characterizing the type of a given task.
        """
        return TaskType.REDUCER

    @abstractmethod
    def reduce_locally(self, samples):
        """
        Converts batch of samples/observations to a numeric value or dictionary.
        For one dataset, this method is executed multiple times, each time for a different data chunk.
        The results from local reductions are then passed directly to the global reduction, which is performed only once.

        Args:
            samples (list): Observation list with all features

        Returns:
            dict,int: Reduced value for a given data chunk
        """
        pass

    @abstractmethod
    def reduce_globally(self, next_local_reduction_gen):
        """
        Finally reduces the entire data set (using locally reduced chunks) to a single numeric value or dictionary.

        Args:
            local_reductions (list): List of locally reduced numbers or dictionaries. Outputs from all locally reduced chunks collected in the list.

        Returns:
            dict,int: Reduced value for a whole dataset
        """
        pass


class AbstractBalancingTask(_AbstractTask):
    """
          Generic abstract class representing task representing a balancing filter adjusting the number of observations to many categories and the number of observations in the category to the selected characteristic.
          This task solves the problem of unbalanced datasets or well-balanced datasets in whic the number of examples should be balanced according to various characteristics (e.g. containing numbers and text that does not contain numbers).
          The word 'category' here does not mean a category during classification task (but it can).
          The category means just an observation category (regardless of the problem being solved), for example,
          a sentence containing a given entity or a sentence without a given entity in a NER task to balance and adjust the number of sentences with a given amount to the number of passes without the given entity.
          One observation may belong to many categories, e.g. balancing the number of observations in the problem of multi-classification.
          This class must be inherited by a specialized class.

        Args:
            max_proportion_difference_category (float): Indicates the maximum difference allowed between the number of observations belonging to different categories.
                The minimum value is 1.0, which means that the number of observations in each category must be equal. (Default 1.0).
            max_proportion_difference_characteristic (float): Indicates the maximum difference allowed between the number of observations belonging to each category.
                This value can influence the number of examples in each category.
                The minimum value is 1.0, which means that the number of observations in categories for each characteristic identified by the value of str will be equal.
                Each category may have finally a different number of observations (depends on the 'max_proportion_difference_category' arg), but if the value of 'max_proportion_difference_characteristic' is 1.0, the proportion of various characteristics for each category will be equal.
                (Default 1.0).
            selection_result_new_column_name (str): The name of the header where the names of the categories to which the given observation has been assigned will be entered to balance the data set.
                An observation can only be assigned to a categories returned by the 'determine_categories' method for the same observation.
                If the 'determine_categories' method returned more than a category, then the given observation may not belong to any category (then it is removed in the same way as in case of filtering tasks),
                or to any other category/categories returned by the 'determine_categories' method. (Default: 'selected_distribution_categories').
            selection_categories_separator (str): Separator which will be used to separate category names in the field defined by the variable 'selection_result_new_column_name' to which observations have been assigned.
        """

    def __init__(self, max_proportion_difference_category=1.0, max_proportion_difference_characteristic=1.0,
                 characteristic_distribution=None,
                 selection_result_new_column_name='selected_distribution_categories',
                 selection_categories_separator=';'):
        self.max_proportion_difference_category = max_proportion_difference_category
        self.max_proportion_difference_characteristic = max_proportion_difference_characteristic
        self.characteristic_distribution=characteristic_distribution
        self.selection_result_new_column_name = selection_result_new_column_name
        self.selection_categories_separator = selection_categories_separator

    def type(self) -> TaskType:
        """
        Returns the type of the given task - balancing task.

        Returns:
        TaskType: Enumerator characterizing the type of a given task.
        """
        return TaskType.BALANCING

    def determine_characteristic(self, sample) -> str:
        """
        Assigns characteristics for a given observation. One observation can only have one characteristic.
        This value is used to balance the number of examples inside each category so that the proportion of examples with different characteristics is not greater than 'max_proportion_difference_characteristic'.
        The method is especially useful when within each category it is required to maintain different types of observations so that the ML algorithm does not learn only one predominant type with a given characteristic,
        for example, keeping the same amount of grayscale and color images for each category while the categories contain several times more color images.
        The default result is the string 'default'. If the same characteristics are returned for each category then balancing by characteristics is turned off because each category contains 100% elements with the same characteristic.

        Args:
            sample (dict): Single observation

        Returns:
            str: Characteristic name. Any text indicating the characteristics of the observation (eg. 'grayscale_obs', 'color_obs', 'text_with_numbers', 'text_without_numbers', etc.
            (Default 'default').
        """
        return 'default'

    @abstractmethod
    def determine_categories(self, sample) -> list:
        """
        Assigns several categories for observation. One observation may have several categories.
        This value is used to balance the number of examples in the dataset so that the proportion of the number of samples in categories does not exceed the 'max_proportion_difference_category' value.
        The method is especially useful when the number of samples in categories is required to be equal to or not greater than the assumed proportional value 'max_proportion_difference_category'.
        For example, keeping the same number of observations for the multiclassification problem, where one observation belongs into multiple categories.
        This method must be signed by the spec class.

        Args:
            sample (dict): Single observation

        Returns:
            list[str]: The list of categories to which the given observation belongs. The category type should be specified with a text value. If a sample belongs to only one category, the method should return a one-element list.
        """
        pass

    @abstractmethod
    def mark_sample_as_selected(self, sample, selected_distribution_categories) -> None:
        """
        The method is called to mark a given data sample as belonging to the selected categories.
        A typical behavior of the method is to create a new field with the name specified in the 'selection_result_new_column_name' variable containing a list of selected categories.

        Args:
            sample (dict): Single observation
            selected_distribution_categories (list[str]): List of categories selected for a given sample to balance the dataset
        """
        sample[self.selection_result_new_column_name] = self.selection_categories_separator.join(
            selected_distribution_categories)
