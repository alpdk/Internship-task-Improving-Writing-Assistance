
from src.metrics.MetricsTemplate import MetricsTemplate

def Accuracy(MetricsTemplate):
    """
    Accuracy metric

    Attributes:
        dataset (pd.DataFrame): dataframe that will be classified
        form_approach (string): formality approach, that will be used for evaluating data
        amount_of_true (int): amount of true labels
        amount_of_false (int): amount of false labels
        true_pos (int): number of true positives results
        true_neg (int): number of true negatives results
        false_pos (int): number of false positives results
        false_neg (int): number of false negatives results
    """

    amount_of_true = 0
    amount_of_false = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    def evaluate_dataset(self):
        """
        This method will provide us a way to evaluate every element from the dataset

        Returns:
            res (float): score of our metrics
        """
        for i in range(self.dataset.shape[0]):
            answer = self.dataset[i]['label']
            generated_answer = self.form_approach.evaluate_sample(self.dataset[i])

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

        return (self.true_pos + self.true_neg) / (self.amount_of_true + self.amount_of_false)
