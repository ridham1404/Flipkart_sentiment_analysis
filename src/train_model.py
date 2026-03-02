import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from preprocess import text_cleaning, tokenize_and_pad
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight



df=pd.read_csv('../data/processed/flipkart_reviews_clean.csv')

label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
y = df['sentiment'].map(label_mapping).values
y=to_categorical(y,num_classes=3)


X_train_text, X_test_text, y_train, y_test = train_test_split(df['Clean_Review'], y, test_size=0.2, random_state=42)
y_indices = np.argmax(y_train, axis=1)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_indices),
    y=y_indices
)

class_weight_dict = dict(enumerate(class_weights))

print(class_weight_dict)
X_train, tokenizer = tokenize_and_pad(X_train_text)
X_test, _ = tokenize_and_pad(X_test_text, tokenizer=tokenizer)




model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    LSTM(64),
     Dropout(0.5),
    Dense(3,activation='softmax')])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)
history=model.fit(
    X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[early_stop],class_weight=class_weight_dict)
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Classification report
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred, target_names=['Negative','Neutral','Positive']))

model.save('models/lstm_model.h5')
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

def predict_sentiment(review):
    review_clean = text_cleaning(review)
    review_pad, _ = tokenize_and_pad([review_clean], tokenizer=tokenizer, max_len=100)
    pred = model.predict(review_pad)
    sentiment_class = np.argmax(pred, axis=1)[0]
    inv_label_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return inv_label_mapping[sentiment_class]

# Example
print(predict_sentiment("The product quality is amazing!"))
print(predict_sentiment("Worst experience ever."))
