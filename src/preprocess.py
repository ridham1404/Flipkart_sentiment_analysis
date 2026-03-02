import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

def text_cleaning(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_pad(texts, tokenizer=None, max_words=5000, max_len=100):
    texts = pd.Series(texts).astype(str).apply(text_cleaning)

    if not tokenizer:
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len)
    return padded, tokenizer