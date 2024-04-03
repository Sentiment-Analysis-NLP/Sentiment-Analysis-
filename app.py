from gui import *
from scrap import *
from setup import *
import pandas as pd
import numpy as np
import spacy
from nltk.tokenize import word_tokenize
from spacytextblob.spacytextblob import SpacyTextBlob


# Download the Punkt tokenizer models.
# Function to extract stock names from a title using SpaCy
def extract_stock_names_spacy(title):
  nlp = spacy.load("en_core_web_sm")
  doc = nlp(title)
  stock_names = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
  stock_names += [
      token.text for token in doc
      if token.pos_ == "PROPN" and token.text.isupper()
  ]
  return stock_names


def get_sentiment(data):
  data['stock_names'] = data['Content'].apply(
      extract_stock_names_spacy) + data['Title'].apply(
          extract_stock_names_spacy)

  spaceit = spacy.load('en_core_web_sm')
  # stb = SpacyTextBlob()
  spaceit.add_pipe('spacytextblob')
  # data['title_subjectivity'] = data['Title'].apply(
  #     lambda x: spaceit(x)._.blob.subjectivity)
  # data['title_polarity'] = data['Title'].apply(
  #     lambda x: spaceit(x)._.blob.polarity)

  data['content_subjectivity'] = data['Content'].apply(
      lambda x: spaceit(x)._.blob.subjectivity)
  data['content_polarity'] = data['Content'].apply(
      lambda x: spaceit(x)._.blob.polarity)
  data.head(5)

  # Save the DataFrame with the new column back to CSV
  data.to_csv('output_file.csv', index=False)

  return data


if __name__ == '__main__':
  # setup()
  # print("Getting data...")
  # data, dataset = Scrape()
  # print("Data Acquired Successfully...")
  # print("Generating sentiment analysis...")
  # output = get_sentiment(data)
  # print("Sentiment analysis complete...")
  Gui()

# Read CSV file
