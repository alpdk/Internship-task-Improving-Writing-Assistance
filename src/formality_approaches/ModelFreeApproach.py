import re
from collections import Counter
from src.formality_approaches.ApproachesTemplate import ApproachesTemplate


class ModelFreeApproach(ApproachesTemplate):
    """
    This class provide formality identification of the text on the based on features related to formal texts

    Attributes:
        formal_score (int): how many times were seen formal attributes in text
        total_features (int): how many features were seen
        words ([str]): list of all words
        text (str): original text
    """

    formal_score = 0
    total_features = 0
    words = []
    text = ""
    name = "ModelFreeApproach"

    def check_contractions(self):
        """
        This method check text on contractions, because formal texts do not contain them.
        """
        contractions = re.findall(r"\b\w+'\w+\b", self.text.lower())

        if contractions:
            self.formal_score -= len(contractions)

        self.total_features += 1

    def check_slang(self):
        """
        This method check text on slang words, because formal texts do not contain them.
        """
        slang_words = {
            'hey', 'hi', 'yo', 'wassup', 'gonna', 'wanna', 'gotta',
            'kinda', 'sorta', 'lemme', 'dunno', 'bout', 'cuz', 'ya'
        }

        slang_count = sum(1 for word in self.words if word in slang_words)

        if slang_count:
            self.formal_score -= slang_count

        self.total_features += 1

    def check_length(self):
        """
        This method check length of the text, because formal texts usually larger.
        """
        sentences = re.split(r'[.!?]', self.text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

            if avg_sentence_length > 12:
                self.formal_score += 1
            elif avg_sentence_length < 8:
                self.formal_score -= 1

            self.total_features += 1

    def check_mult_punct_marks(self):
        """
        This method checks multiple punctuation marks, because formal texts do not contain them.
        """
        exclamation_count = self.text.count('!')

        if exclamation_count > 1:
            self.formal_score -= min(exclamation_count, 3)

        self.total_features += 1

        question_count = self.text.count('?')

        if question_count > 2:
            self.formal_score -= min(question_count, 3)

        self.total_features += 1

    def check_filler_words(self):
        """
        This method check text for filler words, because formal texts do not contain them.
        """
        filler_words = {
            'like', 'um', 'uh', 'ah', 'well', 'so', 'you know', 'i mean',
            'actually', 'basically', 'literally'
        }

        filler_count = sum(1 for word in self.words if word in filler_words)

        if filler_count:
            self.formal_score -= filler_count

        self.total_features += 1

    def check_formal_words(self):
        """
        This method check text for formal words, because formal texts contain them.
        """
        formal_words = {
            'moreover', 'furthermore', 'however', 'therefore', 'thus',
            'consequently', 'nevertheless', 'notwithstanding'
        }

        formal_word_count = sum(1 for word in self.words if word in formal_words)

        if formal_word_count:
            self.formal_score += formal_word_count

        self.total_features += 1

    def check_passive_voice(self):
        """
        This method check text for passive voice structs, because they more often used in formal texts.
        """
        passive_voice = re.findall(r"\b\w+\s\w+\s\w+ed\b", self.text.lower())
        # Simple pattern
        if passive_voice:
            self.formal_score += len(passive_voice)

        self.total_features += 1

    def check_first_person(self):
        """
        This method check text for first person expressions, because they do not appear in formal texts.
        """
        first_person = re.findall(r"\b(I|me|my|mine|we|us|our|ours)\b", self.text, re.IGNORECASE)

        if first_person:
            self.formal_score -= len(first_person) / 2

        self.total_features += 1

    def check_capslock(self):
        """
        This method check text for capslock substrings, because they do not appear in formal texts.
        """
        all_caps = re.findall(r"\b[A-Z]{2,}\b", self.text)

        if all_caps:
            self.formal_score -= len(all_caps)

        self.total_features += 1

    def evaluate_sample(self, text, threshold=0.6):
        """
        This method calculate score by every features and in the end return decision about text formality.

        Parameters:
            text (string): text to check
            threshold (float): The score threshold (0-1) to consider text as formal (default: 0.6)

        Returns:
            answer (bool): is text formal or informal based on linguistic features?
        """

        if not text.strip():
            return False

        self.formal_score = 0
        self.total_features = 0
        self.text = text
        self.words = re.findall(r"\b\w+\b", text.lower())

        self.check_contractions()
        self.check_slang()
        self.check_length()
        self.check_mult_punct_marks()
        self.check_filler_words()
        self.check_formal_words()
        self.check_passive_voice()
        self.check_first_person()
        self.check_capslock()

        normalized_score = (self.formal_score + self.total_features) / (2 * self.total_features)

        return int(normalized_score >= threshold)
