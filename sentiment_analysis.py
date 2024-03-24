import numpy as np
import pandas as pd
import spacy
from textblob import TextBlob

amazon = pd.read_csv('amazon_product_reviews.csv')
amazon.head()

amazon.columns

# Select only the columns that we need

cleaned = amazon[['reviews.text']]
cleaned.head()

# Checking for null values
cleaned.isnull().sum()

cleaned.dropna(inplace = True, axis = 0)

# Obtaining just the reviews
text = cleaned['reviews.text']
text

# load language package
nlp = spacy.load('en_core_web_md')

# Creating a function to preprocess text


def preprocess(text):
    doc = nlp(text.lower().strip())
    processed = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    print(processed)
    
    return ' '.join(processed)

cleaned['processed.text'] = cleaned['reviews.text'].apply(preprocess)

data = cleaned['processed.text'].values
data

for item in data:
    print(item)

def analyze_polarity(text):

# Preprocess the text with spaCy
    doc = nlp(text)
    
# Analyze sentiment with TextBlob
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    return polarity

#Creat a function for sentiment analysis 
sentiments = []

for item in data:
    polarity_score = analyze_polarity(item)  
    
    if polarity_score > 0:
        sentiment = 'positive'
    elif polarity_score < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    sentiments.append(sentiment)

sentiments

#Test the model on sample review

positive_count = sentiments.count('positive')
negative_count = sentiments.count('negative')
neutral_count = sentiments.count('neutral')

total = len(sentiments)

positive_perc = (positive_count / total) * 100
negative_perc = (negative_count / total) * 100
neutral_perc = (neutral_count / total) * 100

print(f"Positive percentage: {positive_perc:.2f}%")
print(f"Negative percentage: {negative_perc:.2f}%")
print(f"Neutral percentage: {neutral_perc:.2f}%")
