
from src.metrics.MetricsTemplate import MetricsTemplate

class F1Score(MetricsTemplate):
    """
    F1Score metric

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

    metric_name = "F1Score"

    def evaluate_dataset(self):
        """
        This method will provide us a way to evaluate every element from the dataset

        Returns:
            res (float): score of our metrics
        """
        for i in range(self.dataset.shape[0]):
            answer = self.dataset.loc[i, 'label']
            generated_answer = self.form_approach.evaluate_sample(self.dataset.loc[i, 'text'])

            if answer == 1:
                self.amount_of_true += 1

                if generated_answer == 1:
                    self.true_pos += 1
                else:
                    self.false_neg += 1
            elif answer == 0:
                self.amount_of_false += 1

                if generated_answer == 0:
                    self.true_neg += 1
                else:
                    self.false_pos += 1

        precision = self.true_pos / (self.true_pos + self.false_pos)
        recall = self.true_pos / (self.true_pos + self.false_neg)

        return 2 * precision * recall / (precision + recall)
