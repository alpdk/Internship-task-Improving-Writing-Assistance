import os
import pandas as pd

from pathlib import Path
from abc import abstractmethod

class MetricsTemplate:
    """
    Template class for metrics

    Attributes:
        dataset (pd.DataFrame): dataframe that will be classified
        form_approach (string): formality approach, that will be used for evaluating data
        amount_of_true (int): amount of true labels
        amount_of_false (int): amount of false labels
        true_pos (int): number of true positives results
        true_neg (int): number of true negatives results
        false_pos (int): number of false positives results
        false_neg (int): number of false negatives results
        metric_name (string): name of the metric
    """

    amount_of_true = 0
    amount_of_false = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    metric_name = None

    def __init__(self, dataset_name, dataset, form_approach):
        """
        Constructor

        Parameters
            dataset_name (string): name of the dataset
            dataset (pd.Dataframe): dataset for evaluation
            form_approach (string): formality approach, that will be used for evaluating data
        """
        self.dataset_name = dataset_name
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

    def evaluate(self):
        """
        This method will save result of specific metric for specific approach
        """
        metric_res = self.evaluate_dataset()
        eval_path = Path(__file__).parent.parent / "metric_results" / f"eval_{self.dataset_name}"

        df = pd.DataFrame()

        if os.path.exists(eval_path):
            df = pd.read_csv(eval_path, index_col=0)

        column_name = self.metric_name
        row_name = self.form_approach.name

        if column_name not in df.columns:
            df[column_name] = 0

        if row_name not in df.index:
            df.loc[row_name] = 0

        df.at[row_name, column_name] = metric_res

        df.to_csv(f"{eval_path}", index=True)
