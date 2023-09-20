import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import os

# Constants for hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
MAX_SEQ_LEN = 100
DROPOUT_RATE = 0.2
PATIENCE = 3

def preprocess_text(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    with open(filename, 'r', encoding='utf-8') as file:
        raw_text = [text.lower().strip() for text in file.read().splitlines() if text]

    questions = raw_text[0::2]
    answers = raw_text[1::2]
     
    for line in raw_text:
        try:
            question, answer = line.split('|||')
            questions.append(question.strip())
            answers.append(answer.strip())
        except:
            pass
    
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(questions + answers)

    vocab_size = len(tokenizer.word_index) + 1

    question_sequences = tokenizer.texts_to_sequences(questions)
    answer_sequences = tokenizer.texts_to_sequences(answers)
    
    padded_questions = pad_sequences(question_sequences, maxlen=MAX_SEQ_LEN, padding='post')
    padded_answers = pad_sequences(answer_sequences, maxlen=MAX_SEQ_LEN, padding='post')

    return tokenizer, vocab_size, padded_questions, padded_answers

def build_model(vocab_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=MAX_SEQ_LEN),
        LSTM(128, return_sequences=True),
        Dropout(DROPOUT_RATE),
        LSTM(128),
        Dense(vocab_size, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, questions, answers):
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    model.fit(questions, np.expand_dims(answers, -1), batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, callbacks=[early_stopping])

def main():
    tokenizer, vocab_size, questions, answers = preprocess_text('dataset.txt')

    model = build_model(vocab_size)
    train_model(model, questions, answers)

    model.save('qa_model.h5')

if __name__ == "__main__":
    main()
