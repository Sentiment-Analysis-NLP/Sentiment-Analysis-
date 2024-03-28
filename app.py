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
    return stock_names

def  get_sentiment(data):
    data['stock_names'] = data['Title'].apply(extract_stock_names_spacy)

    spaceit = spacy.load('en_core_web_sm')
    stb = SpacyTextBlob()
    spaceit.add_pipe(stb)
    data['title_subjectivity'] = data['Title'].apply(lambda x: spaceit(x)._.sentiment.subjectivity)
    data['title_polarity'] = data['Title'].apply(lambda x: spaceit(x)._.sentiment.polarity)


    data['content_subjectivity'] = data['Content'].apply(lambda x: spaceit(x)._.sentiment.subjectivity)
    data['content_polarity'] = data['Content'].apply(lambda x: spaceit(x)._.sentiment.polarity)
    data.head(5)

    # Save the DataFrame with the new column back to CSV
    data.to_csv('output_file.csv', index=False)

    
    return data

if __name__ == '__main__':
    # setup()
    data, dataset = Scrape()
    output = get_sentiment(data)
    
    # Apply the extraction function to the 'title' column
   




# Read CSV file
