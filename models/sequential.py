import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import os

from models.preprocess import process_labeled_data

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

NUM_EPOCHS = 30
VOCAB_SIZE = 10000
EMBEDDING_DIMENSIONS = 8


def read_split_data():
    labeled_data = pd.read_csv(r'/Users/aksharma/PycharmProjects/Trump_Tweets/data/labeled_data.csv')

    preprocessed = process_labeled_data(labeled_data, stemming_lemmatize=True)

    x = preprocessed['text'].to_numpy()
    labels = preprocessed['sentiment'].to_numpy()

    tweets = np.array([' '.join(tweet) for tweet in x])

    return train_test_split(tweets, labels, test_size=0.3, random_state=42, stratify=labels)


def train_tokenizer(train_x, vocab_size):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_x)

    return tokenizer


def vectorize_data(fitted_tokenizer, train_x, test_x):
    max_length = max([len(tweet.split()) for tweet in np.concatenate((train_x, test_x), axis=None)])

    training_sequences = fitted_tokenizer.texts_to_sequences(train_x)
    padding_train = pad_sequences(training_sequences, maxlen=max_length, padding='post')

    testing_sequences = fitted_tokenizer.texts_to_sequences(test_x)
    padding_test = pad_sequences(testing_sequences, maxlen=max_length, padding='post')

    training_padded = np.array(padding_train)
    testing_padded = np.array(padding_test)

    return training_padded, testing_padded, max_length


def create_sequential_nn(vocab_size, embedding_dim, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    return model


def train_fnn(model, train_features, train_labels, test_features, test_labels):
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_features, train_labels, epochs=NUM_EPOCHS, batch_size=250,
                        validation_data=(test_features, test_labels), verbose=1)
    return model, history


def get_model_weights(model):
    return model.layers[0].get_weights()[0]


def plot_accuracy_loss(history, metric_type, epochs):
    metric = history.history[metric_type]
    validation_metric = history.history['val_' + metric_type]

    line = range(1, epochs + 1)

    plt.plot(line, metric, 'bo', label='Train ' + metric_type)
    plt.plot(line, validation_metric, 'b', label='Validation ' + metric_type)

    plt.legend()
    plt.show()


def save_model(trained_model):
    # define model metadata
    MODEL_DIR = "models/demo_model"
    version = 1

    export_path = os.path.join(MODEL_DIR, str(version))
    trained_model.save(export_path, save_format="tf")
    print('\nexport_path = {}'.format(export_path))


def main():
    train_x, test_x, train_y, test_y = read_split_data()

    # fit tokenizer
    fitted_tokenizer = train_tokenizer(train_x, vocab_size=VOCAB_SIZE)

    padded_train, padded_test, max_length = vectorize_data(fitted_tokenizer, train_x, test_x)

    model = create_sequential_nn(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIMENSIONS, max_length=max_length)
    print(model.summary())

    fitted_model, history = train_fnn(model, padded_train, train_y, padded_test, test_y)
    print(history)
    print(fitted_model.evaluate(padded_test, test_y))

    plot_accuracy_loss(history, 'accuracy', NUM_EPOCHS)

    save_model(fitted_model)


if __name__ == '__main__':
    main()
