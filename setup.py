import nltk
import spacy

def setup():
    nltk.download('punkt')
    spacy.cli.download("en_core_web_sm")
    nltk.download('punkt')

