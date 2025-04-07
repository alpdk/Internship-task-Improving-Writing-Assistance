import os
from pathlib import Path

import pandas as pd

from src.metrics.MetricsTemplate import MetricsTemplate
from src.metrics.Accuracy import Accuracy
from src.metrics.Recall import Recall
from src.metrics.Precision import Precision
from src.metrics.F1Score import F1Score


class SumMetric(MetricsTemplate):
    """
    This class will calculate sum of other metrics with some coefficients

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

    metric_name = "SumMetric"

    def evaluate_dataset(self):
        """
        This method will help us to calculate sum metric result

        Returns:
            res (float): score of our metrics
        """

        metrics = [Accuracy(self.dataset_name, self.dataset, self.form_approach),
                   Recall(self.dataset_name, self.dataset, self.form_approach),
                   Precision(self.dataset_name, self.dataset, self.form_approach),
                   F1Score(self.dataset_name, self.dataset, self.form_approach)]

        coef = [0.1, 0.2, 0.4, 0.3]
        res = 0

        eval_path = Path(__file__).parent.parent / "metric_results" / f"eval_{self.dataset_name}"

        df = pd.DataFrame()

        if os.path.exists(eval_path):
            df = pd.read_csv(eval_path, index_col=0)

        for i in range(len(metrics)):
            if (self.metric_name in df.columns) and (self.form_approach.name in df.index) and (
                    df.at[self.form_approach.name, self.metric_name] != 0):
                res += df.at[self.form_approach.name, self.metric_name]
            else:
                res += metrics[i].evaluate_dataset() * coef[i]

        return res
