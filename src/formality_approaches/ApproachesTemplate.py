
from abc import abstractmethod

class ApproachesTemplate:
    """
    This class will help to make some high level wrapper for all approaches

    Attributes
        name (string): name of the approach
    """
    name = None

    @abstractmethod
    def evaluate_sample(self, data):
        """
        Method for classifying the sample.

        Parameters:
            data (string): data that will be classified as formal, or informal

        Returns:
            res (int): 1 if formal sample, 0 if informal sample
        """
        pass