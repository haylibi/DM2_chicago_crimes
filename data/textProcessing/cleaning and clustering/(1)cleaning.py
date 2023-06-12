from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd


def preprocess_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Remove duplicated words
    unique_tokens = list(set(lemmatized_tokens))

    # Sort by size
    unique_tokens.sort(key=len, reverse=True)

    # Join the tokens back into a string
    processed_text = ' '.join(unique_tokens)
    return processed_text

#this txt was generated from selecting unique Location Description and unique Description
with open('description.txt', 'r') as file:
#with open('location.txt', 'r') as file:
    descriptions = file.readlines()

# Preprocess and reduce location descriptions
preprocessed_descriptions = [preprocess_text(desc.strip()) for desc in descriptions]

# Preprocess and reduce location descriptions
preprocessed_descriptions = [preprocess_text(desc) for desc in descriptions]

#this is the result from cleaning the words from Location and Descriptions txt file
#This result will be used in the (2)clustering
#file = open('location_clean.txt','w')
file = open('description_clean.txt','w')
file.writelines(preprocessed_descriptions)
file.close()
