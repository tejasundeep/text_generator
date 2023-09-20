import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import os

# Constants for hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
MAX_SEQ_LEN = 100
DROPOUT_RATE = 0.2
PATIENCE = 3

# Read and preprocess the text dataset
def preprocess_text(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")
    
    with open(filename, 'r', encoding='utf-8') as file:
        raw_text = [text.lower().strip() for text in file.read().splitlines() if text]

    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(raw_text)
    vocab_size = len(tokenizer.word_index) + 1

    text_sequences = tokenizer.texts_to_sequences(raw_text)
    padded_sequences = pad_sequences(text_sequences, maxlen=MAX_SEQ_LEN - 1, padding='post', truncating='post')

    target_data = [seq[1:] + [0] for seq in text_sequences]
    target_data = pad_sequences(target_data, maxlen=MAX_SEQ_LEN - 1, padding='post')

    return tokenizer, vocab_size, padded_sequences, np.array(target_data)

# Build LSTM model
def build_model(vocab_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=MAX_SEQ_LEN - 1),
        LSTM(128, return_sequences=True),
        Dropout(DROPOUT_RATE),
        BatchNormalization(),
        LSTM(128, return_sequences=True),
        Dropout(DROPOUT_RATE),
        BatchNormalization(),
        LSTM(128, return_sequences=True),
        Dropout(DROPOUT_RATE),
        BatchNormalization(),
        LSTM(128),
        Dropout(DROPOUT_RATE),
        BatchNormalization(),
        Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, padded_sequences, target_data):
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    model.fit(padded_sequences, target_data, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, callbacks=[early_stopping])

# Text generation function
def generate_text(model, tokenizer, initial_text, max_length):
    generated_text = initial_text
    for _ in range(max_length):
        initial_seq = tokenizer.texts_to_sequences([generated_text])[0]
        padded_initial = pad_sequences([initial_seq], maxlen=MAX_SEQ_LEN - 1, padding='post', truncating='post')
        pred_probs = model.predict(padded_initial)[0]
        pred_id = np.argmax(pred_probs)
        pred_word = tokenizer.index_word.get(pred_id, '<unknown>')
        generated_text += " " + pred_word
    return generated_text

# Main function
def main():
    tokenizer, vocab_size, padded_sequences, target_data = preprocess_text('dataset.txt')

    model = build_model(vocab_size)
    train_model(model, padded_sequences, target_data)

    # Save the model
    model.save('text_generation_model.h5')
    
    seed_text = "once upon a time"
    generated_text = generate_text(model, tokenizer, seed_text, max_length=50)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()
