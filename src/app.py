# src/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from preprocess import text_cleaning, tokenize_and_pad
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pkl")
MAX_LEN = 100  # same as used during training
inv_label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}


model = load_model(MODEL_PATH)
print("Model loaded from:", MODEL_PATH)
print("Model summary:")
model.summary()
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)


app = FastAPI(title="Flipkart Sentiment Analysis API")


class Review(BaseModel):
    review: str


@app.get("/")
def home():
    return {"message": "Flipkart Sentiment Analysis API is running!"}


@app.post("/predict")
def predict_sentiment(data: Review):
    clean_text = text_cleaning(data.review)
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        seq,
        maxlen=MAX_LEN
    )

    pred = model.predict(padded)

    pred_class = np.argmax(pred, axis=1)[0]
    sentiment = inv_label_mapping[pred_class]

    return {"review": data.review, "sentiment": sentiment}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1",port=8000)


