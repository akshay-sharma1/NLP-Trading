import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import os

from models.preprocess import process_labeled_data

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# model hyperparameters
NUM_EPOCHS = 30
VOCAB_SIZE = 10000
EMBEDDING_DIMENSIONS = 8


def read_split_data():
    """
        Reads the labeled data csv file and preprocesses it with the routine from
        preprocess.py. Transforms each tweet into a normal string and maps both
        the tweet data and the labels into a numpy array. Applies the train_test_split
        routine from sklearn in order to split into train and validation sets.

            args: None
            ret:
                train_x: train tweets
                test_x: test tweets
                train_y: train labels
                test_y: test labels

    """
    labeled_data = pd.read_csv(r'/Users/aksharma/PycharmProjects/Trump_Tweets/data/labeled_data.csv')

    preprocessed = process_labeled_data(labeled_data, stemming_lemmatize=True)

    x = preprocessed['text'].to_numpy()
    labels = preprocessed['sentiment'].to_numpy()

    tweets = np.array([' '.join(tweet) for tweet in x])

    return train_test_split(tweets, labels, test_size=0.3, random_state=42, stratify=labels)


def train_tokenizer(train_x, vocab_size):
    """
        Trains a sentence tokenizer on the train tweets. This tokenizer is used
        in order to transform both the train and test sets into vectorized one-hot
        encodings.

            args:
                train_x: train set
                vocab_size: max amount of vocab "tokens" or features
            ret:
                tokenizer: trained tokenizer on vocab from train set

    """
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_x)

    return tokenizer


def vectorize_data(fitted_tokenizer, train_x, test_x):
    """
        Transforms both the train and test data into one-hot encodings
        based on the vocab features from the fitted_tokenizer. Resulting
        encodings are padded to a max length that is calculated based on
        the length of the longest tweet. Out of vocab tokens are represented
        with the index {'<OOV>': 1}.

            args:
                fitted_tokenizer: tokenizer that has been trained on a train set
                train_x: train tweets
                test_x: validation set tweets
            ret:
                training_padded: vectorized train data
                testing_padded: vectorized test data
                max_length: max length of one tweet

    """
    max_length = max([len(tweet.split()) for tweet in np.concatenate((train_x, test_x), axis=None)])

    training_sequences = fitted_tokenizer.texts_to_sequences(train_x)
    padding_train = pad_sequences(training_sequences, maxlen=max_length, padding='post')

    testing_sequences = fitted_tokenizer.texts_to_sequences(test_x)
    padding_test = pad_sequences(testing_sequences, maxlen=max_length, padding='post')

    training_padded = np.array(padding_train)
    testing_padded = np.array(padding_test)

    return training_padded, testing_padded, max_length


def create_sequential_nn(vocab_size, embedding_dim, max_length):
    """
        Creates a sequential neural network with 4 total layers and using
        the input hyperparameters.

            args:
                vocab_size: total number of vocab features
                embedding_dim: length of word embeddings
                max_length: max length of any tweet
            ret:
                model: sequential neural network

    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    return model


def train_fnn(model, train_features, train_labels, test_features, test_labels):
    """
        Trains the sequential neural network using rmsprop as the optimizer and
        sparse_categorical_crossentropy (probabilities) as the loss function.

            args:
                model: sequential model
                train_features: vectorized train data
                test_features: vectorized test data
            ret:
                model: the now fitted model on the train data
                history: documentation of accuracy/other metrics as the model
                        fits on the number of epochs.

    """
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_features, train_labels, epochs=NUM_EPOCHS, batch_size=250,
                        validation_data=(test_features, test_labels), verbose=1)
    return model, history


def plot_accuracy_loss(history, metric_type, epochs):
    """
        Plots either the accuracy and loss over time using the history
        object derived from model training.

            args:
                history: sequential model
                metric_type: either 'loss' or 'accuracy'
                epochs: total number of epochs for training
            ret:
                None: plt is shown

    """
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

    # vectorize using tokenizer
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
