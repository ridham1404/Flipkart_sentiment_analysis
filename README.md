# Flipkart Sentiment Analysis 

An end-to-end Deep Learning based Sentiment Analysis system built using LSTM and deployed with FastAPI.

## Features

- 3-class sentiment classification (Negative, Neutral, Positive)
- Handles severe class imbalance using computed class weights
- Trained on 150k+ Flipkart reviews
- REST API built using FastAPI
- Real-time prediction endpoint

##  Model Architecture

- Embedding Layer (5000 vocab size, 128 dimensions)
- LSTM (64 units)
- Dropout (0.5)
- Dense (Softmax)

Total Parameters: ~689k

## Performance

Test Accuracy: ~90%

| Class     | Precision | Recall | F1-score |
|-----------|----------|--------|----------|
| Negative  | 0.60     | 0.99   | 0.75     |
| Neutral   | 0.98     | 0.85   | 0.91     |
| Positive  | 1.00     | 0.90   | 0.95     |

##  How to Run

```bash
pip install -r requirements.txt
uvicorn src.app:app --reload