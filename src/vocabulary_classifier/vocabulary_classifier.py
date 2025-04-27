import json
import nltk
from nltk import RSLPStemmer

class VocabularyClassifier:

    def __init__(self):
        with open(r'vocabulary_classifier\parameters.json', 'r') as openfile:
            json_object = json.load(openfile)
        self.CLASSIFICATION_THRESHOLD = json_object["CLASSIFICATION_THRESHOLD"]
        self.DICTIONARY = json_object["DICTIONARY"]
        nltk.download('rslp')
        self.stemmer = RSLPStemmer()

    def transform(self, word):
        return self.stemmer.stem(word)       

    def news_classifier(self, sentence: str):
        words = [self.transform(word) for word in sentence.split()]

        score = 0
        for word in words:
            score += self.DICTIONARY.get(word,0)
        if score > self.CLASSIFICATION_THRESHOLD:
            return "FAKE"
        else:
            return "REAL"
        
vocab_classifier = VocabularyClassifier()