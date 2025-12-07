# Sentiment Analysis Using LSTMs

## Project Overview
This project implements a sentiment analysis system using **Long Short-Term Memory (LSTM)** neural networks. It processes text data (tweets about New Year’s resolutions) to classify sentiment into three categories: **positive, neutral, negative**.

### Key Features
- Text preprocessing: cleaning, tokenization, stopword removal.
- Sentiment labeling with **TextBlob polarity**.
- Sequence conversion with **Keras Tokenizer**.
- LSTM model with embeddings and dropout.
- Evaluation with confusion matrix & classification report.

## Workflow
1. Data Preparation: Load dataset, clean text, assign sentiment labels.
2. Preprocessing: Tokenize and pad sequences, encode labels.
3. Model Architecture:
- Embedding layer (50k vocab, 128-dim)
- LSTM (128 units, dropout 0.5)
- Dense softmax output (3 classes)
4. Training: Early stopping, 5 epochs, batch size 64.
5. - Evaluation:
- Accuracy: ~80%
- Positive class: strong precision/recall
- Negative class: good recall
- Neutral class: weaker recall (0.48)
Results
- Confusion Matrix:
[[217  14  37]
[ 41  76  41]
[ 40  28 509]]

- Classification Report:
- Negative: Precision 0.73, Recall 0.81, F1 0.77
- Neutral: Precision 0.64, Recall 0.48, F1 0.55
- Positive: Precision 0.87, Recall 0.88, F1 0.87

Prediction System
def predict_sentiment(review):
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=50)
    pred = model.predict(padded).argmax(axis=-1)[0]
    return ['negative','neutral','positive'][pred]

Future Improvements
- Handle class imbalance (neutral underrepresented).
- Try bidirectional LSTMs.
- Use pretrained embeddings (Word2Vec, GloVe).
- Experiment with transformer-based models (BERT, RoBERTa).


