import torch

from transformers import pipeline
from huggingface_hub import login
from src.formality_approaches.ApproachesTemplate import ApproachesTemplate

class HuggingFaceModelApproach(ApproachesTemplate):
    """
    This class implements the model approach for identification of formal and informal texts

    Attributes:
        name (string): name of the approach
        model (): model to use
    """

    def __init__(self, model_name, huggingface_token):
        """
        Constructor

        Parameters:
            model_name (string): name of the model from the HuggingFace
            huggingface_token (string): HuggingFace token for work with model
        """
        self.name = model_name

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        login(token=huggingface_token)
        self.model = pipeline('text-generation', model=model_name, device=device, trust_remote_code=True)

    def evaluate_sample(self, sample):
        """
        Method for classifying the sample.

        Parameters:
            sample (string): data that will be classified as formal, or informal

        Returns:
            res (int): 1 if formal sample, 0 if informal sample
        """

        # prompt = f"Read this text and tell is it formal or not: {sample}. If it is formal, print only 1, otherwise only 0."
        prompt = sample
        model_chain = self.model(prompt, max_length=2, truncation=True)
        response = model_chain[0]['generated_text']

        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return int(response)
