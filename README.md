# Sentiment-Analysis
Sentiment Analysis with LSTM – My hands‑on deep learning project where I apply Long Short‑Term Memory networks to classify text sentiment, showcasing how preprocessing and sequence modeling can uncover emotions in language.

## Overview
This project trains an **LSTM neural network** to classify text sentiment into **positive, neutral, or negative** using New Year’s resolution tweets.


## Workflow
- Preprocess text (clean, tokenize, remove stopwords).
- Label sentiment with TextBlob polarity.
- Convert text to padded sequences.
- Train LSTM model with embeddings.
- Evaluate (~80% accuracy).

Usage
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('sentiment_model.h5')
tokenizer = pickle.load(open('tokenizer.pkl','rb'))

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=50)
    pred = model.predict(padded).argmax(axis=-1)[0]
    return ['negative','neutral','positive'][pred]

print(predict_sentiment("I want to study in the new year."))
Output: positive