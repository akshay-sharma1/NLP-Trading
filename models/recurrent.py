import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as plt

from preprocess import process_labeled_data
from models.subword_encoder import BPE

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

NUM_EPOCHS = 10


def read_preprocess():
    unprocessed = pd.read_csv('/Users/aksharma/PycharmProjects/Trump_Tweets/data/labeled_data.csv')

    preprocessed = process_labeled_data(unprocessed, stemming_lemmatize=False)

    x = preprocessed['text'].to_numpy()
    labels = preprocessed['sentiment'].to_numpy()

    tweets = np.array([' '.join(tweet) for tweet in x])

    return tweets, labels


def vectorize_data(train_data, test_data):
    pass


def create_recurrent_nn(vocab_size, embedding_dim, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])


def train_rnn(model, train_features, train_labels, test_features, test_labels):
    pass


def plot_accuracy_loss(model, history):
    pass


def main():
    tweets, _ = read_preprocess()

    encoder = BPE(tweets, num_merges=1000)
    encoder.train_vocab()

    encoded = encoder.encode_word('gughy')
    print(encoded)


if __name__ == '__main__':
    main()
