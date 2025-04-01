import pandas as pd

from abc import abstractmethod

class MetricsTemplate:
    """
    Template class for metrics

    Attributes:
        dataset (pd.DataFrame): dataframe that will be classified
        form_approach (string): formality approach, that will be used for evaluating data
    """

    def __init__(self, dataset: pd.DataFrame, form_approach):
        """
        Constructor

        Parameters
            dataset (pd.DataFrame): dataframe that will be classified
            form_approach (string): formality approach, that will be used for evaluating data
        """
        self.dataset = dataset
        self.form_approach = form_approach

    @abstractmethod
    def evaluate_dataset(self):
        """
        This method will provide us a way to evaluate every element from the dataset

        Returns:
            res (float): score of our metrics
        """
        pass

