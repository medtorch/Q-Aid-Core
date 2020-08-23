import re
import nltk
import string
from words import greeting_inputs, definition_inputs, vqa_filter

nltk.download("punkt")
nltk.download("wordnet")


class NLP:
    def __init__(self):
        self.wnlemmatizer = nltk.stem.WordNetLemmatizer()
        self.punctuation_removal = dict(
            (ord(punctuation), None) for punctuation in string.punctuation
        )

    def perform_lemmatization(self, tokens):
        return [self.wnlemmatizer.lemmatize(token) for token in tokens]

    def get_processed_text(self, document):
        return self.perform_lemmatization(
            nltk.word_tokenize(document.lower().translate(self.punctuation_removal))
        )

    def is_greeting(self, words, query):
        if query in greeting_inputs:
            return True
        for word in words:
            if word in greeting_inputs:
                return True
        return False

    def is_definition(self, words, query):
        for sep in definition_inputs:
            if sep in query:
                return query.split(sep)[1]
        return None

    def is_vqa_safe(self, words, query):
        for word in words:
            if word in vqa_filter:
                return True
        return False

    def ask(self, query):
        words = self.get_processed_text(query)
        query = " ".join(words)

        if self.is_greeting(words, query):
            return {"type": "greeting"}

        define = self.is_definition(words, query)
        if define:
            return {
                "type": "wiki",
                "data": "".join(self.get_processed_text(define)),
            }
        if self.is_vqa_safe(words, query):
            return {
                "type": "vqa",
            }
        return {
            "type": "invalid",
        }
