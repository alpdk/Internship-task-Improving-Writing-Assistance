import google.generativeai as genai

from src.formality_approaches.ApproachesTemplate import ApproachesTemplate


class GeminiApproach(ApproachesTemplate):
    """
    This class uses Google's Gemini 2.0 Flash model to classify text formality.

    Attributes:
        name (string): name of the approach
        model: loaded Gemini model instance
    """
    name = "Gemini2.0Flash"

    def __init__(self, model_name, api_key):
        """
        Initialize Gemini model with API key.

        Parameters:
            model_name (string): name of the model
            api_key (string): API key
        """
        if not api_key:
            raise ValueError("Gemini API key not provided in args")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(model_name)

    def evaluate_sample(self, sample):
        """
        Classify text formality using Gemini.

        Parameters:
            sample (string): text to be classified

        Returns:
            res (int): 1 if formal, 0 if informal
        """
        prompt = f"""
        Analyze the following text and determine if it is formal or informal. 
        Respond ONLY with '1' for formal or '0' for informal. No other text or explanation.

        Text: "{sample}"

        Answer:
        """

        try:
            response = self.model.generate_content(prompt)

            classification = response.text.strip()

            if classification not in ['0', '1']:
                raise ValueError(f"Unexpected response from Gemini: {response.text}")

            return int(classification)

        except Exception as e:
            print(f"Error classifying sample with Gemini: {e}")
            return 0
