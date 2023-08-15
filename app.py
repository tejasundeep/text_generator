import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

# Constants for hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
MAX_SEQ_LEN = 100
DROPOUT_RATE = 0.2
PATIENCE = 3

# Read and preprocess the text dataset
def preprocess_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        raw_text = [text.lower().strip() for text in file.read().splitlines()]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(raw_text)
    vocab_size = len(tokenizer.word_index) + 1

    text_sequences = tokenizer.texts_to_sequences(raw_text)
    padded_sequences = pad_sequences(text_sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

    return tokenizer, vocab_size, padded_sequences

# Build LSTM model
def build_model(vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=MAX_SEQ_LEN))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(DROPOUT_RATE))
    model.add(BatchNormalization())
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(DROPOUT_RATE))
    model.add(BatchNormalization())
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(DROPOUT_RATE))
    model.add(BatchNormalization())
    model.add(LSTM(128))
    model.add(Dropout(DROPOUT_RATE))
    model.add(BatchNormalization())
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, padded_sequences, target_data):
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    model.fit(padded_sequences, target_data, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, callbacks=[early_stopping])

# Text generation function
def generate_text(model, tokenizer, initial_text, max_length):
    generated_text = initial_text
    for _ in range(max_length):
        initial_seq = tokenizer.texts_to_sequences([initial_text])[0]
        padded_initial = pad_sequences([initial_seq], maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
        pred_probs = model.predict(padded_initial)[0]
        pred_id = np.argmax(pred_probs)
        pred_word = tokenizer.index_word.get(pred_id, '<unknown>')
        generated_text += " " + pred_word
        initial_text = generated_text
    return generated_text

# Main function
def main():
    tokenizer, vocab_size, padded_sequences = preprocess_text('dataset.txt')
    target_data = tf.keras.utils.to_categorical(padded_sequences, num_classes=vocab_size)

    model = build_model(vocab_size)
    train_model(model, padded_sequences, target_data)

    seed_text = "once upon a time"
    generated_text = generate_text(model, tokenizer, seed_text, max_length=50)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()
